from addict import Dict

# 0.468452B
gptconfig_nano = Dict()
gptconfig_nano.hidden_size = 1024
gptconfig_nano.num_attention_heads = 16
gptconfig_nano.vocab_size = 56000
gptconfig_nano.dropout_prob = 0.01
gptconfig_nano.max_position_embeddings = 1024
gptconfig_nano.num_layers = 32
gptconfig_nano.ignore_index = -1
gptconfig_nano.flash_attention = True

# 6.705648B
gptconfig_base = Dict()
gptconfig_base.hidden_size = 4096
gptconfig_base.num_attention_heads = 32
gptconfig_base.vocab_size = 56000
gptconfig_base.dropout_prob = 0.1
gptconfig_base.max_position_embeddings = 1024
gptconfig_base.num_layers = 32
gptconfig_base.ignore_index = -1
gptconfig_base.flash_attention = True


gptconfigs = {
    'nano': gptconfig_nano,
    'base': gptconfig_base,
}
