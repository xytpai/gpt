import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from dacite import from_dict
from typing import Optional
from torch import Tensor


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.a_weight = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.b_weight = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
    
    def forward(self, x):
        dtype = x.dtype
        x = self.alpha * (x.float() @ self.a_weight @ self.b_weight)
        return x.to(dtype)


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


class AttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.dropout_prob = config.dropout_prob
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.num_query_groups = config.num_query_groups
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=True, dtype=config.dtype)
        kv_size = self.attention_head_size * self.num_query_groups
        self.key = nn.Linear(config.hidden_size, kv_size, bias=True, dtype=config.dtype)
        self.value = nn.Linear(config.hidden_size, kv_size, bias=True, dtype=config.dtype)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=config.dtype)
        self.kv_cache = None
        # lora
        self.query_lora = LoRALayer(config.hidden_size, config.hidden_size, config.lora_rank, config.lora_alpha)
        self.key_lora = LoRALayer(config.hidden_size, kv_size, config.lora_rank, config.lora_alpha)
        self.value_lora = LoRALayer(config.hidden_size, kv_size, config.lora_rank, config.lora_alpha)
        self.dense_lora = LoRALayer(config.hidden_size, config.hidden_size, config.lora_rank, config.lora_alpha)

    @staticmethod
    def gen_rope_cache(max_seq_length: int, n_elem: int, 
        dtype: torch.dtype, device: torch.device, base: int = 10000):
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))
        seq_idx = torch.arange(max_seq_length, dtype=dtype, device=device)
        idx_theta = torch.outer(seq_idx, theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache # -> [seq_length, n_elem, 2]

    def apply_rotary_emb(self, x, rope_cache):
        batch_size, seq_length, num_heads, head_size = x.size()
        rot_dim = rope_cache.shape[-2] * 2
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        xshaped = x.reshape(batch_size, seq_length, num_heads, rot_dim // 2, 2)
        rope_cache = rope_cache.view(1, seq_length, 1, xshaped.size(3), 2)
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out2 = x_out2.flatten(3)
        return torch.cat((x_out2, x_pass), dim=-1)

    def forward(self, x, attention_mask, rope_cache, input_pos: Optional[Tensor]=None, xa: Optional[Tensor]=None):
        batch_size, seq_length, hidden_size = x.size()
        q = self.query(x) + self.query_lora(x)
        x_for_kv = x if xa is None else xa
        k = self.key(x_for_kv) + self.key_lora(x_for_kv)
        v = self.value(x_for_kv) + self.value_lora(x_for_kv)
        q = q.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        k = k.view(batch_size, seq_length, self.num_query_groups, self.attention_head_size)
        v = v.view(batch_size, seq_length, self.num_query_groups, self.attention_head_size)
        if rope_cache is not None:
            q = self.apply_rotary_emb(q, rope_cache)
            k = self.apply_rotary_emb(k, rope_cache)
        q, k, v = [item.transpose(1, 2).contiguous() for item in [q, k, v]] # -> (b, nh, t, hs)
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)
        k = k.repeat_interleave(self.num_attention_heads // self.num_query_groups, dim=1)
        v = v.repeat_interleave(self.num_attention_heads // self.num_query_groups, dim=1)
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=self.dropout_prob)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        output = self.dense(output) + self.dense_lora(output)
        return output


class FeedForwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.ffn_hidden_size * 2, bias=False, dtype=config.dtype)
        self.dense_4h_to_h = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False, dtype=config.dtype)

    def forward(self, x):
        x = self.dense_h_to_4h(x)
        x = torch.chunk(x, 2, dim=-1)
        x = F.silu(x[0]) * x[1]
        x = self.dense_4h_to_h(x)
        return x


class MOEFeedForwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(config.num_experts, config.ffn_hidden_size, config.hidden_size))
        self.w2 = nn.Parameter(torch.empty(config.num_experts, config.hidden_size, config.ffn_hidden_size))
        self.w3 = nn.Parameter(torch.empty(config.num_experts, config.ffn_hidden_size, config.hidden_size))
        self.hidden_size = config.hidden_size
        self.num_activated_experts = config.num_activated_experts

    def forward(self, x):
        x = x.view(-1, self.hidden_size)
        expert_weights = F.softmax(self.gate(x), dim=-1) # _, num_experts
        expert_weights, expert_indices = torch.topk(expert_weights, self.num_activated_experts, dim=-1)
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True) # _, num_activated_experts
        x1 = F.silu(torch.einsum('ti, taoi -> tao', 
                                 x, self.w1[expert_indices])) # _, num_activated_experts, ffn_hidden_size
        x3 = torch.einsum('ti, taoi -> tao', 
                          x, self.w3[expert_indices]) # _, num_activated_experts, ffn_hidden_size
        expert_outs =  torch.einsum('tao, taio -> tai', (x1 * x3), 
                                    self.w2[expert_indices]) # _, num_activated_experts, hidden_size
        return torch.einsum('tai,ta -> ti', expert_outs, expert_weights) # _, hidden_size


class RMSNormLayer(nn.Module):
    def __init__(self, config, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(config.hidden_size, dtype=config.dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = AttentionLayer(config)
        if config.enable_visual:
            self.visual_attention = AttentionLayer(config)
            self.visual_norm = RMSNormLayer(config)
            self.enable_visual = True
        else:
            self.enable_visual = False
        if config.enable_audio:
            self.audio_attention = AttentionLayer(config)
            self.audio_norm = RMSNormLayer(config)
            self.enable_audio = True
        else:
            self.enable_audio = False
        self.feed_forward = FeedForwardLayer(config)
        self.input_norm = RMSNormLayer(config)
        self.post_attention_norm = RMSNormLayer(config)

    def forward(self, x, attention_mask, rope_cache, input_pos: Optional[Tensor]=None, xa: Optional[Tensor]=None):
        x = x + self.attention(self.input_norm(x), attention_mask, rope_cache, input_pos)
        if self.enable_visual:
            x = x + self.visual_attention(self.visual_norm(x), attention_mask, rope_cache, input_pos, xa)
        if self.enable_audio:
            x = x + self.audio_attention(self.audio_norm(x), attention_mask, rope_cache, input_pos, xa)
        out = x + self.feed_forward(self.post_attention_norm(x))
        return out


class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, self.hidden_size, dtype=config.dtype)

    def forward(self, input_ids):
        embeddings = self.word_embeddings(input_ids)
        return embeddings


class GPTXModel(nn.Module):

    @dataclass
    class Config:
        hidden_size: int
        ffn_hidden_size: int
        num_attention_heads: int
        vocab_size: int
        dropout_prob: float
        num_layers: int
        num_query_groups: int
        ignore_index: int
        dtype: torch.dtype
        max_seq_length: int
        enable_visual: bool
        enable_audio: bool
        lora_rank: int
        lora_alpha: float

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings_name = 'word_embeddings_' + str(config.vocab_size)
        setattr(self, self.word_embeddings_name, EmbeddingLayer(config))
        self.rotary_dim = config.hidden_size // config.num_attention_heads
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNormLayer(config)
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.dtype)
        self.max_seq_length = self.config.max_seq_length
        self.register_buffer('causal_mask', torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)), persistent=False)
        self.rope_cache = None

    def device(self):
        return self.output_layer.weight.device
    
    def dtype(self):
        return self.config.dtype
    
    def size(self):
        return sum([np.prod(list(p.size())) for p in self.parameters()])
    
    def assign_kv_cache(self, max_batch_size):
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(
                max_batch_size, self.max_seq_length, self.config.num_query_groups, 
                self.config.hidden_size // self.config.num_attention_heads,
                self.dtype(), self.device())
    
    def forward(self, input_ids, input_pos: Optional[Tensor]=None, 
        images: Optional[Tensor]=None, target_ids: Optional[Tensor]=None, temperature=0.9):
        # input_ids: L[b, t]
        # images: F[b, c, h, w]
        if self.rope_cache is None:
            self.rope_cache = self.__gen_rope_cache()
        _, seq_length = input_ids.size()
        word_embeddings_obj = getattr(self, self.word_embeddings_name)
        embeddings = word_embeddings_obj(input_ids) # [b, t, hidden_size]
        if input_pos is None:
            input_pos = torch.arange(0, seq_length, device=self.device())
            rope_cache = self.rope_cache[input_pos]
            causal_mask = self.causal_mask[None, None, :seq_length, :seq_length]
        else: # KV Cache enabled
            rope_cache = self.rope_cache[input_pos]
            causal_mask = self.causal_mask[None, None, input_pos]
        for layer in self.layers:
            embeddings = layer(embeddings, causal_mask, rope_cache, input_pos)
        embeddings = self.final_norm(embeddings)
        if target_ids is None:
            return self.__post_pred(embeddings, temperature)
        else:
            return self.__post_loss(embeddings, target_ids)
    
    def __gen_rope_cache(self):
        rope_cache = AttentionLayer.gen_rope_cache(
            self.max_seq_length, self.rotary_dim // 2, self.dtype(), self.device())
        return rope_cache

    def __post_loss(self, embeddings, target_ids):
        embeddings = self.output_layer(embeddings) # -> F[b, t, vocab_size]
        loss = F.cross_entropy(embeddings.view(-1, embeddings.size(-1)), 
            target_ids.contiguous().view(-1), 
            ignore_index=self.config.ignore_index)
        return loss

    def __post_pred(self, embeddings, temperature):
        embeddings = self.output_layer(embeddings[:, -1, :]) # -> F[b, vocab_size]
        embeddings = embeddings / temperature
        ids_next = torch.multinomial(F.softmax(embeddings, dim=-1), num_samples=1) # b
        return ids_next


class ChatApplication:
    def __init__(self, model, tokenizer, compile=False):
        if not compile:
            self.model = model
            self.enable_flash = True
        else:
            self.model = torch.compile(model)
            self.enable_flash = False
        self.tokenizer = tokenizer
        self.max_seq_length = self.model.max_seq_length

    @torch.no_grad()
    def generate(self, input_text, temperature=0.9):
        self.model.assign_kv_cache(1)
        input_ids = self.tokenizer.encode(input_text, bos=True, eos=True)
        T = len(input_ids)
        prompt = torch.empty([1, self.max_seq_length], device=self.model.device(), dtype=torch.long)
        input_pos = torch.arange(0, T, device=self.model.device())
        prompt[:, input_pos] = torch.LongTensor(input_ids).to(self.model.device()).view(1, -1)
        latency = []
        for i in range(self.max_seq_length - T):
            torch.cuda.synchronize()
            start = time.time()
            with torch.backends.cuda.sdp_kernel(
                enable_flash=self.enable_flash, enable_mem_efficient=self.enable_flash, enable_math=True):
                idx_next = self.model(prompt[:, input_pos], input_pos, temperature=temperature)
            torch.cuda.synchronize()
            end = time.time()
            latency.append(end - start)
            idx_next_item = idx_next[:, -1].item()
            word_next = self.tokenizer.decode([idx_next_item])
            print(word_next, end='', flush=True)
            if idx_next_item == self.tokenizer.eos_id:
                break
            input_pos = torch.tensor([T], device=self.model.device(), dtype=torch.int)
            prompt[:, T] = idx_next
            T += 1
        latency = latency[10:]
        token_per_s = 1.0 / (sum(latency) / len(latency))
        print(f'\ntoken_per_s:{token_per_s}')
        return prompt


if __name__ == '__main__':
    import sys
    from configs import get_gpt_config
    from tokenization import SPTokenizer

    wpath = sys.argv[1]
    input_text = sys.argv[2] # "写一篇一万字短片科幻小说，要求讲述人类探索亚空间的故事"

    config = from_dict(data_class=GPTXModel.Config, data=get_gpt_config('6b'))
    print(config)
    model = GPTXModel(config)

    ckpt = torch.load(wpath, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print('load model: ' + str({'missing_keys':missing_keys, 'unexpected_keys':unexpected_keys}))

    model = model.cuda()
    type_size = 1
    print(model)
    print('Model {} : params: {:4f}B'.format(model._get_name(), model.size() * type_size / 1000 / 1000 / 1000))

    tokenizer = SPTokenizer('tokenizer.model')
    chatapp = ChatApplication(model, tokenizer)
    output = chatapp.generate(input_text)
