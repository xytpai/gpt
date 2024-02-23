import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import time
from dataclasses import dataclass
from dacite import from_dict


# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton_unique_kernel_names = True


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_size, dtype, device):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_size)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, input_pos: int, k_val, v_val):
        # k_val, v_val: [b, nh, t, hs]
        k_out = self.k_cache
        v_out = self.v_cache
        end = input_pos + k_val.shape[2]
        k_out[:, :, input_pos:end, :] = k_val
        v_out[:, :, input_pos:end, :] = v_val
        return k_out[:, :, :end], v_out[:, :, :end]


class AttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.dropout_prob = config.dropout_prob
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.num_query_groups = config.num_query_groups
        qkv_hsize = config.hidden_size + 2 * self.attention_head_size * self.num_query_groups
        self.qkv = nn.Linear(config.hidden_size, qkv_hsize, bias=True, dtype=config.dtype)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=config.dtype)
        self.kv_cache = None

    @staticmethod
    def gen_rope_cache(seq_length: int, n_elem: int, 
        dtype: torch.dtype, device: torch.device, base: int = 10000):
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))
        seq_idx = torch.arange(seq_length, dtype=dtype, device=device)
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
        rope_cache = rope_cache[:seq_length].view(1, seq_length, 1, xshaped.size(3), 2)
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out2 = x_out2.flatten(3)
        return torch.cat((x_out2, x_pass), dim=-1)

    def forward(self, x, attention_mask, rope_cache, input_pos: int=0):
        batch_size, seq_length, hidden_size = x.size()
        q, k, v = self.qkv(x).split([
            hidden_size, 
            self.num_query_groups * self.attention_head_size,
            self.num_query_groups * self.attention_head_size], dim=-1)
        q = q.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        k = k.view(batch_size, seq_length, self.num_query_groups, self.attention_head_size)
        v = v.view(batch_size, seq_length, self.num_query_groups, self.attention_head_size)
        if rope_cache is not None:
            q = self.apply_rotary_emb(q, rope_cache)
            k = self.apply_rotary_emb(k, rope_cache)
        q, k, v = [item.transpose(1, 2).contiguous() for item in [q, k, v]] # -> (b, nh, t, hs)
        # print(q.shape, q.stride())
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)
        k = k.repeat_interleave(self.num_attention_heads // self.num_query_groups, dim=1)
        v = v.repeat_interleave(self.num_attention_heads // self.num_query_groups, dim=1)
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=self.dropout_prob, is_causal=(seq_length>1))
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        output = self.dense(output)
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
        self.feed_forward = FeedForwardLayer(config)
        self.input_norm = RMSNormLayer(config)
        self.post_attention_norm = RMSNormLayer(config)

    def forward(self, x, attention_mask, rope_cache, input_pos: int=0):
        h = x + self.attention(self.input_norm(x), attention_mask, rope_cache, input_pos)
        out = h + self.feed_forward(self.post_attention_norm(h))
        return out


class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, self.hidden_size, dtype=config.dtype)

    def forward(self, input_ids):
        embeddings = self.word_embeddings(input_ids)
        return embeddings


class GPTModel(nn.Module):

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

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings_name = 'word_embeddings_' + str(config.vocab_size)
        setattr(self, self.word_embeddings_name, EmbeddingLayer(config))
        self.rotary_dim = config.hidden_size // config.num_attention_heads
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNormLayer(config)
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.dtype)
    
    def device(self):
        return self.output_layer.weight.device
    
    def dtype(self):
        return self.config.dtype
    
    def assign_kv_cache(self, max_batch_size, max_seq_length):
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.config.num_query_groups, 
                self.config.hidden_size // self.config.num_attention_heads,
                self.dtype(), self.device())
    
    def gen_rope_cache(self, max_seq_length):
        rope_cache = AttentionLayer.gen_rope_cache(
            max_seq_length, self.rotary_dim // 2, self.dtype(), self.device())
        return rope_cache

    def forward(self, rope_cache, input_ids, input_pos: int=0):
        # input_ids: L[b, t]
        _, seq_length = input_ids.size()
        word_embeddings_obj = getattr(self, self.word_embeddings_name)
        embeddings = word_embeddings_obj(input_ids) # [b, t, hidden_size]
        rope_cache = rope_cache[input_pos:]
        
        for layer in self.layers:
            embeddings = layer(embeddings, None, rope_cache, input_pos)
        embeddings = self.final_norm(embeddings)
        return embeddings

    def post_loss(self, embeddings, target_ids):
        embeddings = self.output_layer(embeddings) # -> F[b, t, vocab_size]
        loss = F.cross_entropy(embeddings.view(-1, embeddings.size(-1)), 
            target_ids.contiguous().view(-1), 
            ignore_index=self.config.ignore_index)
        return loss

    def post_pred(self, embeddings):
        embeddings = self.output_layer(embeddings[:, -1, :]) # -> F[b, vocab_size]
        return embeddings


class ChatApplication:
    def __init__(self, model, tokenizer, compile=True):
        if not compile:
            self.model = model
        else:
            '''
  File "/tmp/torchinductor_xytpai/jx/cjxhaaobiiwit5hrsx6y5ygwxpm3obfrt46xw5srhd2vlq2ctiek.py", line 1400, in call
    buf17 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf8, (1, 32, 1, 128), (4096, 128, 1, 1), 0), reinterpret_tensor(buf15, (1, 32, 7, 128), (0, 896, 128, 1), 0), reinterpret_tensor(buf16, (1, 32, 7, 128), (0, 896, 128, 1), 0), None, False)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/xytpai/miniconda3/envs/xyt/lib/python3.11/site-packages/torch/_ops.py", line 692, in __call__
    return self._op(*args, **kwargs or {})
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  RuntimeError: query is not correctly aligned (strideM)
            '''
            self.model = torch.compile(model)
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(self, input_text, temperature=1.0, max_len=2000):
        self.model.assign_kv_cache(1, 2048)
        rope_cache = self.model.gen_rope_cache(2048)
        input_ids = self.tokenizer.encode(input_text, bos=True, eos=True)
        input_ids = torch.LongTensor(input_ids).to(self.model.device()).view(1, -1)
        input_pos = 0
        output_text = []
        latency = []
        for i in range(max_len):
            torch.cuda.synchronize()
            start = time.time()
            embeddings = self.model(rope_cache, input_ids, input_pos)
            output = self.model.post_pred(embeddings)
            torch.cuda.synchronize()
            end = time.time()
            dur = (end - start)
            latency.append(dur)
            output = output / temperature
            idx_next = torch.multinomial(F.softmax(output, dim=-1), num_samples=1) # b
            idx_next_item = idx_next[:, -1].item()
            word_next = self.tokenizer.decode([idx_next_item])
            print(word_next, end='', flush=True)
            output_text.append(word_next)
            if idx_next_item == self.tokenizer.eos_id:
                break
            input_pos += input_ids.shape[1]
            input_ids = idx_next
        
        latency = latency[10:]
        token_per_s = 1.0 / (sum(latency) / len(latency))
        print(f'\ntoken_per_s:{token_per_s}')

        return "".join(output_text)


if __name__ == '__main__':
    from configs import get_gpt_config
    from tokenization import SPTokenizer

    config = from_dict(data_class=GPTModel.Config, data=get_gpt_config('6b'))
    model = GPTModel(config)

    ckpt = torch.load('weight.ckpt', map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print('load model: ' + str({'missing_keys':missing_keys, 'unexpected_keys':unexpected_keys}))

    model = model.cuda()
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 1
    print(config)
    print(model)
    print('Model {} : params: {:4f}B'.format(model._get_name(), para * type_size / 1000 / 1000 / 1000))

    tokenizer = SPTokenizer('tokenizer.model')
    chatapp = ChatApplication(model, tokenizer)
    input_text = "北京有什么好玩的地方"
    output = chatapp.generate(input_text)
    # print(output)
