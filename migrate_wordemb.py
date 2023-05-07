import argparse
import torch
import torch.nn as nn
from tokenization import Tokenizer


def load_param(args):
    ckpt = torch.load(args.ckpt, map_location='cpu')
    args.state = ckpt
    ckpt = ckpt['model']
    print(ckpt.keys())
    param = ckpt[args.param]
    return param


def load_tokenizer(args):
    new_tk = Tokenizer()
    old_tk = Tokenizer(args.old_vocab)
    return new_tk, old_tk


def migrate(args, new_tk, old_tk, param):
    new_vocab = new_tk.tokenizer.get_vocab()
    old_vocab = old_tk.tokenizer.get_vocab()

    new_vocab_size, emb_size = len(new_vocab.keys()), param.shape[1]
    new_emb = nn.Embedding(new_vocab_size, emb_size)
    torch.nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
    new_emb.weight.requires_grad = False

    count = 0
    for key, idx in new_vocab.items():
        if old_vocab.get(key, None) is not None:
            new_emb.weight[idx, :] = param[old_vocab[key], :]
            # print(key + ' ' + str(old_idx) + '->' + str(new_idx))
            count += 1
    print(str(count) + ' migrated')
    print('writting ...')
    args.state['model'][args.param][:] = new_emb.weight
    torch.save(args.state, args.ckpt + '.new')


def parse_args():
    parser = argparse.ArgumentParser(description='Migrate word embeddings')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--param', type=str, required=True)
    parser.add_argument('--old_vocab', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    param = load_param(args)
    print(args.param, param.shape, param.dtype)
    new_tokenizer, old_tokenizer = load_tokenizer(args)
    migrate(args, new_tokenizer, old_tokenizer, param)


if __name__ == '__main__':
    main()
