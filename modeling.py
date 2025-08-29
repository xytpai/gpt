import os
import time
import json
import tiktoken
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from dacite import from_dict
from torch import Tensor
from typing import Optional
from tokenization import Tokenizer, Message, ChatFormat


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 32
    max_seq_len: int = 2048
    dropout_prob: float = 0.0
    dtype: torch.dtype = torch.half
    # MoE
    num_experts: int = 0
    num_activated_experts: int = 0
    # Training
    ignore_index: int = -1


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_size, dtype, device):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_size)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, input_pos: Tensor, k_val, v_val):
        # input_pos: L[t]
        # k_val, v_val: F[b, nh, t, hs]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.dim % args.n_heads == 0
        self.dropout_prob = args.dropout_prob
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.n_rep = args.n_heads // self.n_kv_heads
        # weights
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False, dtype=args.dtype)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=args.dtype)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=args.dtype)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False, dtype=args.dtype)
        self.kv_cache = None

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis
    
    def apply_rotary_emb(self, xq, xk, freqs_cis):
        def reshape_for_broadcast(freqs_cis, x):
            ndim = x.ndim
            assert 0 <= 1 < ndim
            assert freqs_cis.shape == (x.shape[1], x.shape[-1])
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
            return freqs_cis.view(*shape)
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(self, x, attention_mask, freqs_cis, input_pos: Optional[Tensor]=None, xa: Optional[Tensor]=None):
        batch_size, seq_length, _ = x.size()
        # Infer xq, xk and xv
        xq = self.wq(x)
        x_for_kv = x if xa is None else xa
        xk = self.wk(x_for_kv)
        xv = self.wv(x_for_kv)
        xq = xq.view(batch_size, seq_length, -1, self.head_dim)
        xk = xk.view(batch_size, seq_length, -1, self.head_dim)
        xv = xv.view(batch_size, seq_length, -1, self.head_dim)
        # Apply RoPE
        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis)
        # Refine xq, xk and xv shape
        xq, xk, xv = [item.transpose(1, 2).contiguous() for item in [xq, xk, xv]] # -> (b, nh, t, hs)
        if self.kv_cache is not None:
            xk, xv = self.kv_cache.update(input_pos, xk, xv)
        xk = xk.repeat_interleave(self.n_rep, dim=1)
        xv = xv.repeat_interleave(self.n_rep, dim=1)
        # DSPA
        output = F.scaled_dot_product_attention(
            xq, xk, xv, attn_mask=attention_mask, dropout_p=self.dropout_prob)
        # Infer output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        return self.wo(output)


def refine_hidden_dim(hidden_dim, multiple_of=256, ffn_dim_multiplier=None):
    hidden_dim = int(2 * hidden_dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        hidden_dim = refine_hidden_dim(4 * args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False, dtype=args.dtype)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False, dtype=args.dtype)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False, dtype=args.dtype)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MOEFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        hidden_dim = refine_hidden_dim(4 * args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.gate = nn.Linear(args.dim, args.num_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(args.num_experts, hidden_dim, args.dim))
        self.w2 = nn.Parameter(torch.empty(args.num_experts, args.dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(args.num_experts, hidden_dim, args.dim))
        self.dim = args.dim
        self.num_activated_experts = args.num_activated_experts

    def forward(self, x):
        x = x.view(-1, self.dim)
        expert_weights = F.softmax(self.gate(x), dim=-1) # _, num_experts
        expert_weights, expert_indices = torch.topk(expert_weights, self.num_activated_experts, dim=-1)
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True) # _, num_activated_experts
        x1 = F.silu(torch.einsum('ti, taoi -> tao', 
                                 x, self.w1[expert_indices])) # _, num_activated_experts, hidden_dim
        x3 = torch.einsum('ti, taoi -> tao', 
                          x, self.w3[expert_indices]) # _, num_activated_experts, hidden_dim
        expert_outs =  torch.einsum('tao, taio -> tai', (x1 * x3), 
                                    self.w2[expert_indices]) # _, num_activated_experts, dim
        return torch.einsum('tai,ta -> ti', expert_outs, expert_weights) # _, dim


class RMSNorm(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.eps = args.norm_eps
        self.weight = torch.nn.Parameter(torch.ones(args.dim, dtype=args.dtype))

    def forward(self, x):
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = x.to(input_dtype)
        return self.weight * x


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        if getattr(args, 'enable_visual', None):
            self.visual_attention = Attention(args)
            self.visual_norm = RMSNorm(args)
            self.enable_visual = True
        else:
            self.enable_visual = False
        if getattr(args, 'enable_audio', None):
            self.audio_attention = Attention(args)
            self.audio_norm = RMSNorm(args)
            self.enable_audio = True
        else:
            self.enable_audio = False
        self.layer_id = layer_id
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args)
        self.ffn_norm = RMSNorm(args)

    def forward(self, x, attention_mask, freqs_cis, input_pos: Optional[Tensor]=None, xa: Optional[Tensor]=None):
        x = x + self.attention(self.attention_norm(x), attention_mask, freqs_cis, input_pos)
        if self.enable_visual:
            x = x + self.visual_attention(self.visual_norm(x), attention_mask, freqs_cis, input_pos, xa)
        if self.enable_audio:
            x = x + self.audio_attention(self.audio_norm(x), attention_mask, freqs_cis, input_pos, xa)
        out = x + self.feed_forward(self.ffn_norm(x))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.max_seq_len = args.max_seq_len
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim, dtype=args.dtype)
        self.layers = nn.ModuleList([TransformerBlock(layer_id, args) for layer_id in range(args.n_layers)])
        self.norm = RMSNorm(args)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False, dtype=args.dtype)
        self.freqs_cis = Attention.precompute_freqs_cis(
            args.dim // args.n_heads,
            self.max_seq_len * 2,
            args.rope_theta,
        )
        self.register_buffer('causal_mask', torch.tril(
            torch.ones(self.max_seq_len, self.max_seq_len, dtype=torch.bool)), persistent=False)

    def device(self):
        return self.output.weight.device
    
    def dtype(self):
        return self.args.dtype
    
    def size(self):
        return sum([np.prod(list(p.size())) for p in self.parameters()])
    
    def assign_kv_cache(self, max_batch_size):
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(
                max_batch_size, self.max_seq_len, self.args.n_kv_heads, 
                self.args.dim // self.args.n_heads,
                self.dtype(), self.device())
    
    def forward(
        self, 
        tokens, 
        input_pos: Optional[Tensor]=None, 
        images: Optional[Tensor]=None):
        # tokens: L[b, t]
        # images: F[b, c, h, w]
        _, seq_length = tokens.size()
        h = self.tok_embeddings(tokens) # [b, t, hidden_size]
        self.freqs_cis = self.freqs_cis.to(self.device())

        if input_pos is None:
            input_pos = torch.arange(0, seq_length, device=self.device())
            freqs_cis = self.freqs_cis[input_pos]
            causal_mask = self.causal_mask[None, None, :seq_length, :seq_length]
        else: # KV Cache enabled
            freqs_cis = self.freqs_cis[input_pos]
            causal_mask = self.causal_mask[None, None, input_pos]
        for layer in self.layers:
            h = layer(h, causal_mask, freqs_cis, input_pos)
        h = self.norm(h)
        return self.output(h)

    @staticmethod
    def post_loss(h, target_ids):
        # h -> F[b, t, vocab_size]
        loss = F.cross_entropy(h.view(-1, h.size(-1)), 
            target_ids.contiguous().view(-1), 
            ignore_index=self.args.ignore_index)
        return loss

    @staticmethod
    def post_pred(h, temperature):
        h = h[:, -1, :] / temperature # -> F[b, vocab_size]
        ids_next = torch.multinomial(F.softmax(h, dim=-1), num_samples=1) # b
        return ids_next


class SimpleChatApp:
    def __init__(self, file, compile=False):
        self.load_model(file)
        if not compile:
            self.enable_flash = True
        else:
            self.model = torch.compile(self.model)
            self.enable_flash = False
        self.max_seq_len = self.model.max_seq_len
    
    def load_model(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            jf = json.load(f)
        args = from_dict(data_class=ModelArgs, data=jf)
        print(args)
        model = Transformer(args)
        state_dict = {}
        for wfile in jf['wfiles']:
            ckpt = torch.load(wfile, map_location='cpu')
            state_dict.update(ckpt)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print('load model: ' + str({'missing_keys':missing_keys, 'unexpected_keys':unexpected_keys}))
        type_size = 1
        print('Model {} : params: {:4f}B'.format(model._get_name(), model.size() * type_size / 1000 / 1000 / 1000))
        self.tokenizer = Tokenizer(jf['tfile'])
        self.formatter = ChatFormat(self.tokenizer)
        self.model = model.cuda()

    @torch.no_grad()
    def generate(self, input_text, temperature=0.9):
        message = {
            "role": "user",
            "content": input_text,
        }
        self.model.assign_kv_cache(1)
        input_ids = self.formatter.encode_message(message)
        T = len(input_ids)
        prompt = torch.empty([1, self.max_seq_len], device=self.model.device(), dtype=torch.long)
        input_pos = torch.arange(0, T, device=self.model.device())
        prompt[:, input_pos] = torch.LongTensor(input_ids).to(self.model.device()).view(1, -1)
        latency = []
        for i in range(self.max_seq_len - T):
            torch.cuda.synchronize()
            start = time.time()
            # with torch.backends.cuda.sdp_kernel(
            #     enable_flash=self.enable_flash, enable_mem_efficient=self.enable_flash, enable_math=True):
            if True:
                h = self.model(prompt[:, input_pos], input_pos)
                idx_next = Transformer.post_pred(h, temperature)
            torch.cuda.synchronize()
            end = time.time()
            latency.append(end - start)
            idx_next_item = idx_next[:, -1].item()
            word_next = self.tokenizer.decode([idx_next_item])
            if idx_next_item in self.tokenizer.stop_tokens:
                break
            print(word_next, end='', flush=True)
            input_pos = torch.tensor([T], device=self.model.device(), dtype=torch.int)
            prompt[:, T] = idx_next
            T += 1
        latency = latency[10:]
        token_per_s = 1.0 / (sum(latency) / len(latency))
        print(f'\ntoken_per_s:{token_per_s}')
        return prompt


if __name__ == '__main__':
    import sys
    cpath = sys.argv[1]
    input_text = sys.argv[2]
    chatapp = SimpleChatApp(cpath)
    output = chatapp.generate(input_text)
