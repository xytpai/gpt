from addict import Dict


gptconfig_nano = Dict()
gptconfig_nano.hidden_size = 256
gptconfig_nano.num_attention_heads = 16
gptconfig_nano.vocab_size = 119547
gptconfig_nano.dropout_prob = 0.1
gptconfig_nano.max_position_embeddings = 128
gptconfig_nano.num_layers = 12
gptconfig_nano.ignore_index = 0


gptconfig_base = Dict()
gptconfig_base.hidden_size = 512
gptconfig_base.num_attention_heads = 16
gptconfig_base.vocab_size = 119547
gptconfig_base.dropout_prob = 0.1
gptconfig_base.max_position_embeddings = 256
gptconfig_base.num_layers = 24
gptconfig_base.ignore_index = 0


gptconfigs = {
    'nano': gptconfig_nano,
    'base': gptconfig_base,
}
