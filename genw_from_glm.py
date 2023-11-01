import os
import sys
import glob
import torch

indir = sys.argv[1]
outdir = sys.argv[2]
assert outdir.endswith('.ckpt')
infiles = sorted(glob.glob(indir + '*.bin'))

ckpt = {}
for infile in infiles:
    ckpt.update(torch.load(os.path.join(indir, infile), map_location='cpu'))

new_ckpt ={}
for key, value in ckpt.items():
    new_key = key
    
    new_key = new_key.replace('transformer.embedding.word_embeddings.', 'word_embeddings_65024.word_embeddings.')

    new_key = new_key.replace('transformer.encoder.layers.', 'layers.')
    new_key = new_key.replace('input_layernorm.', 'input_norm.')
    new_key = new_key.replace('self_attention.query_key_value.', 'attention.qkv.')
    new_key = new_key.replace('self_attention.dense.', 'attention.dense.')
    new_key = new_key.replace('post_attention_layernorm.', 'post_attention_norm.')
    new_key = new_key.replace('mlp.dense_h_to_4h.', 'feed_forward.dense_h_to_4h.')
    new_key = new_key.replace('mlp.dense_4h_to_h.', 'feed_forward.dense_4h_to_h.')

    new_key = new_key.replace('transformer.encoder.final_layernorm.', 'final_norm.')

    print(f'{key} -> {new_key} : {list(value.shape)} {value.dtype}')
    new_ckpt[new_key] = value

def save_ckpt(ckpt, savedir):
    torch.save(ckpt, savedir)

save_ckpt(new_ckpt, outdir)
