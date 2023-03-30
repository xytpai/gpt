import argparse
import torch

import modeling
import tokenization
from configs import gptconfigs
from train import load_model


class Inferencer(object):
    def __init__(self, args, model):
        self.tokenizer = tokenization.FullTokenizer('vocab.txt')
        self.args = args
        self.model = model
        self.model.eval()

    def pred(self, text):
        tokens = self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.LongTensor(ids).view(1, -1).cuda()
        with torch.no_grad():
            ids_o = self.model.generate(ids, self.args.max_position_embeddings * 2)
            ids_o = list(ids_o.cpu()[0].numpy())
            ids_inv = self.tokenizer.convert_ids_to_tokens(ids_o)
            ids_inv = [x for x in ids_inv if x != '[UNK]']
            print(''.join(ids_inv))


def main(args):
    config = modeling.GPTConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        vocab_size=args.vocab_size,
        dropout_prob=args.dropout_prob,
        max_position_embeddings=args.max_position_embeddings,
        num_layers=args.num_layers,
        ignore_index=args.ignore_index,
    )
    model = modeling.GPT(config)
    load_model(args, model)
    model = model.cuda(args.devices[0])
    model.eval()
    infer = Inferencer(args, model)
    infer.pred(args.text)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT Evaluating')
    parser.add_argument('--model', type=str, default='nano')
    parser.add_argument('--devices', type=list, default=[0])
    parser.add_argument('--text', type=str, default='')
    args = parser.parse_args()
    new_args = gptconfigs[args.model]
    new_args.update(vars(args))
    main(new_args)
