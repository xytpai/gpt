import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from dacite import from_dict


@dataclass
class GPTConfig:
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    vocab_size: int
    dropout_prob: float
    num_layers: int
    num_query_groups: int
    ignore_index: int
    dtype: torch.dtype


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
        seq_length, batch_size, num_heads, head_size = x.size()
        rot_dim = rope_cache.shape[-2] * 2
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        xshaped = x.reshape(seq_length, batch_size, num_heads, rot_dim // 2, 2)
        rope_cache = rope_cache[:seq_length].view(seq_length, 1, 1, xshaped.size(3), 2)
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out2 = x_out2.flatten(3)
        return torch.cat((x_out2, x_pass), dim=-1)

    def forward(self, x, attention_mask, rope_cache):
        seq_length, batch_size, hidden_size = x.size()
        q, k, v = self.qkv(x).split([
            hidden_size, 
            self.num_query_groups * self.attention_head_size,
            self.num_query_groups * self.attention_head_size], dim=-1)
        q = q.view(seq_length, batch_size, self.num_attention_heads, self.attention_head_size)
        k = k.view(seq_length, batch_size, self.num_query_groups, self.attention_head_size)
        v = v.view(seq_length, batch_size, self.num_query_groups, self.attention_head_size)
        if rope_cache is not None:
            q = self.apply_rotary_emb(q, rope_cache)
            k = self.apply_rotary_emb(k, rope_cache)
        k = k.unsqueeze(-2).expand(-1, -1, -1, self.num_attention_heads // self.num_query_groups, -1)
        k = k.contiguous().view(seq_length, batch_size, self.num_attention_heads, self.attention_head_size)
        v = v.unsqueeze(-2).expand(-1, -1, -1, self.num_attention_heads // self.num_query_groups, -1)
        v = v.contiguous().view(seq_length, batch_size, self.num_attention_heads, self.attention_head_size)
        q, k, v = [item.permute(1, 2, 0, 3) for item in [q, k, v]] # -> (b, nh, t, hs)
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=self.dropout_prob, is_causal=True)
        output = output.permute(2, 0, 1, 3).contiguous().view(seq_length, batch_size, hidden_size)
        return self.dense(output)


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

    def forward(self, x, attention_mask, rope_cache):
        h = x + self.attention(self.input_norm(x), attention_mask, rope_cache)
        out = h + self.feed_forward(self.post_attention_norm(h))
        return out


class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, self.hidden_size, dtype=config.dtype)

    def forward(self, input_ids):
        embeddings = self.word_embeddings(input_ids)
        embeddings = embeddings.transpose(0, 1).contiguous() # -> [s, b, h]
        return embeddings
    

class GPTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings_name = 'word_embeddings_' + str(config.vocab_size)
        setattr(self, self.word_embeddings_name, EmbeddingLayer(config))
        self.rotary_dim = config.hidden_size // config.num_attention_heads
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNormLayer(config)
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.dtype)
    
    def get_word_embeddings_weight(self):
        word_embeddings_obj = getattr(self, self.word_embeddings_name)
        return word_embeddings_obj.word_embeddings.weight

    def forward(self, input_ids, attention_mask):
        _, seq_length = input_ids.size()
        word_embeddings_obj = getattr(self, self.word_embeddings_name)
        embeddings = word_embeddings_obj(input_ids) # (t, b, hidden_size)
        rope_cache = AttentionLayer.gen_rope_cache(
            seq_length, self.rotary_dim // 2, embeddings.dtype, embeddings.device)
        for layer in self.layers:
            embeddings = layer(embeddings, attention_mask, rope_cache)
        embeddings = self.final_norm(embeddings)
        embeddings = self.output_layer(embeddings)
        return embeddings


class ChatApplication:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    @torch.no_grad()
    def generate(self, input_ids):
        out_text = []
        for i in range(50):
            output = self.model(input_ids, None)
            temperature = 1.0
            output = output[-1, :, :] / temperature
            idx_next = torch.multinomial(F.softmax(output, dim=-1), num_samples=1) # b
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            text =self.tokenizer.decode([idx_next.item()])
            out_text.append(text)
            print(text, end='', flush=True)
        return out_text
    

if __name__ == '__main__':
    from configs import get_gpt_config
    from tokenization import SPTokenizer

    config = from_dict(data_class=GPTConfig, data=get_gpt_config('6b'))
    model = GPTLayer(config)

    ckpt = torch.load('weight.ckpt', map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print('load model: ' + str({'missing_keys':missing_keys, 'unexpected_keys':unexpected_keys}))
 
    model = model.cuda()
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 1
    print('Model {} : params: {:4f}B'.format(model._get_name(), para * type_size / 1000 / 1000 / 1000))
    print(config)
    print(model)

    # fake_input = torch.rand(2, 10) * 100  # batch_size, seq_length
    tokenizer = SPTokenizer('tokenizer.model')
    chatapp = ChatApplication(model, tokenizer)
    input_text = '已知一块钱等于3个馒头，问十二块钱等于什么？'
    input_ids = tokenizer.encode(input_text, bos=False, eos=False)
    input_ids = torch.LongTensor(input_ids).view(1, -1).cuda()
    output = chatapp.generate(input_ids)
    print(output)

    # fake_input = fake_input.long().cuda()
    # pred = model(fake_input, None)
    # print(pred)
