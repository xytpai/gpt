import torch
from addict import Dict


# 6.243584B
gpt_config_6b = Dict()
gpt_config_6b.hidden_size = 4096
gpt_config_6b.ffn_hidden_size = 13696
gpt_config_6b.num_attention_heads = 32
gpt_config_6b.vocab_size = 65024
gpt_config_6b.dropout_prob = 0.0
gpt_config_6b.num_layers = 28
gpt_config_6b.num_query_groups = 2
gpt_config_6b.ignore_index = -1
gpt_config_6b.dtype = torch.float16
gpt_config_6b.max_seq_length = 2048


def get_gpt_config(name):
    if name == '6b':
        return gpt_config_6b
    raise KeyError(f"Config {name} not found")
