import math
from dataclasses import dataclass
import re
from dacite import from_dict
import torch
import torch.nn as nn
from torch.nn import functional as F
from ext import RMSNorm


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(freqs_cis) # -> FloatTensor(end, dim//2, 2)


def apply_rotary_emb(xq, xk, freqs_cis):
    # batch_size, nhead, t, hidden_size = xq.shape
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    freqs_cis = torch.view_as_complex(freqs_cis)
    xq_out = torch.view_as_real(torch.view_as_complex(xq_) * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(torch.view_as_complex(xk_) * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


@dataclass
class GPTConfig:
    hidden_size: int
    num_attention_heads: int
    vocab_size: int
    dropout_prob: float
    max_position_embeddings: int
    num_layers: int
    ignore_index: int
    flash_attention: bool


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of=256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.flash = config.flash_attention
        self.dropout_prob = config.dropout_prob
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.wq = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.wk = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.wv = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        if not self.flash:
            self.attention_dropout = nn.Dropout(config.dropout_prob)
            self.register_buffer('bias', torch.tril(
                torch.ones(config.max_position_embeddings, config.max_position_embeddings)
                    ).view(1, 1, config.max_position_embeddings, config.max_position_embeddings),
                persistent=False)

    def forward(self, x, freqs_cis):
        batch_size, seq_length, hidden_size = x.size()
        q = self.wq(x).view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2) # -> (b, nh, t, hs)
        k = self.wk(x).view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2) # -> (b, nh, t, hs)
        v = self.wv(x).view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2) # -> (b, nh, t, hs)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        if self.flash:
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_prob, is_causal=True)
        else:
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.attention_head_size) # -> (b, nh, t, t)
            attention_scores = attention_scores.masked_fill(self.bias[:,:,:seq_length,:seq_length] == 0, float('-inf'))
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.attention_dropout(attention_probs) # -> (b, nh, t, t)
            output = torch.matmul(attention_probs, v) # -> (b, nh, t, hs)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        return self.wo(output)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CausalAttention(config)
        self.feed_forward = FeedForward(dim=config.hidden_size, hidden_dim=4*config.hidden_size)
        self.attention_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)

    def forward(self, x, freqs_cis):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings_name = 'word_embeddings_' + str(config.vocab_size)
        setattr(self, self.word_embeddings_name, nn.Embedding(config.vocab_size, config.hidden_size))
        freqs_cis = precompute_freqs_cis(config.hidden_size//config.num_attention_heads, config.max_position_embeddings*2)
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('attention_out_layer.2.weight') or pn.endswith('dense.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None, start_pos=0):
        _, seq_length = input_ids.size()
        assert seq_length <= self.config.max_position_embeddings
        word_embeddings_obj = getattr(self, self.word_embeddings_name)
        embeddings = word_embeddings_obj(input_ids) # (b, t, hidden_size)
        freqs_cis = self.freqs_cis.float()
        for layer in self.layers:
            embeddings = layer(embeddings, freqs_cis[start_pos:start_pos+seq_length])
        embeddings = self.norm(embeddings)
        if targets is not None:
            output = F.linear(embeddings, word_embeddings_obj.weight, None) # b, t, vocab_size
            loss = F.cross_entropy(output.view(-1, output.size(-1)), 
                targets.view(-1), ignore_index=self.config.ignore_index)
        else:
            output = F.linear(embeddings[:, [-1], :], word_embeddings_obj.weight, None) # b, 1, vocab_size
            loss = None
        return output, loss

    @torch.no_grad()
    def generate(self, tokenizer, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        eos = tokenizer.text_to_ids('[EOS]')[0]
        len_input = input_ids.numel()
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at max_position_embeddings
            if input_ids.size(1) <= self.config.max_position_embeddings:
                idx_cond = input_ids
            else:
                idx_cond = input_ids[:, -self.config.max_position_embeddings:]
            # forward the model to get the logits for the index in the sequence
            output, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            output = output[:, -1, :] / temperature # b, vocab_size
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(output, min(top_k, output.size(-1)))
                output[output < v[:, [-1]]] = -float('Inf')
            idx_next = torch.multinomial(F.softmax(output, dim=-1), num_samples=1) # b

            # print
            raw = tokenizer.ids_to_text([idx_next])
            if(len(re.findall(r'\b[A-Za-z]+\b', raw))) > 0:
                raw = raw + ' '
            print(raw, end='', flush=True)

            if idx_next.item() == eos:
                break
                # pass
            # append sampled index to the running sequence and continue
            input_ids = torch.cat((input_ids, idx_next), dim=1)
        return input_ids[:, len_input:]


class RelationGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.score_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, targets=None, start_pos=0):
        # targets: LongTensor (batch_size)
        batch_size, seq_length = input_ids.size()
        assert seq_length <= self.config.max_position_embeddings
        word_embeddings_obj = getattr(self, self.word_embeddings_name)
        embeddings = word_embeddings_obj(input_ids) # (b, t, hidden_size)
        for block in self.blocks:
            embeddings = block(embeddings, self.freqs_cis[start_pos:start_pos+seq_length])
        embeddings = self.layernorm(embeddings)
        output = self.score_head(embeddings[:, [-1], :]).view(batch_size)
        if targets is not None:
            loss = F.binary_cross_entropy_with_logits(output, targets)
        else:
            output = output.sigmoid()
            loss = None
        return output, loss


if __name__ == '__main__':
    from configs import gptconfig_nano, gptconfig_base
    import numpy as np
    config = from_dict(data_class=GPTConfig, data=gptconfig_nano)
    model = GPT(config).cpu()
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 1
    print('Model {} : params: {:4f}B'.format(model._get_name(), para * type_size / 1000 / 1000 / 1000))
    # print(model)
    fake_input = torch.rand(2, 10) * 100  # batch_size, seq_length
    fake_input = fake_input.long().cpu()
    pred, _ = model(fake_input)
    print(pred)
    masked_lm_labels = torch.randn(2, 10) * 100
    masked_lm_labels = masked_lm_labels.long().cpu()
    masked_lm_labels[masked_lm_labels < 50] = config.ignore_index
    pred, loss = model(fake_input, masked_lm_labels)
    print(loss)
    loss.backward()
    # idxs = model.generate(fake_input, 20, 1.0, 100)
    # print(idxs.shape, idxs)

    # RelationGPT
    model = RelationGPT(config).cpu()
    pred, _ = model(fake_input)
    print(pred)
    relation_labels = torch.rand(2)
    pred, loss = model(fake_input, relation_labels)
    print(loss)
    loss.backward()
