from addict import Dict

# 0.46B
gptconfig_nano = Dict()
gptconfig_nano.hidden_size = 768
gptconfig_nano.num_attention_heads = 12
gptconfig_nano.vocab_size = 48000
gptconfig_nano.dropout_prob = 0.1
gptconfig_nano.max_position_embeddings = 1024
gptconfig_nano.num_layers = 32
gptconfig_nano.ignore_index = -1
gptconfig_nano.flash_attention = True

# 2.5B
gptconfig_base = Dict()
gptconfig_base.hidden_size = 2048
gptconfig_base.num_attention_heads = 32
gptconfig_base.vocab_size = 48000
gptconfig_base.dropout_prob = 0.1
gptconfig_base.max_position_embeddings = 1024
gptconfig_base.num_layers = 64
gptconfig_base.ignore_index = -1
gptconfig_base.flash_attention = True


gptconfigs = {
    'nano': gptconfig_nano,
    'base': gptconfig_base,
}
