import argparse
import torch

import modeling
import tokenization


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
            ids_o = self.model.generate(ids, 128)
            ids_o = list(ids_o.cpu()[0].numpy())
            ids_inv = self.tokenizer.convert_ids_to_tokens(ids_o)
            print(ids_inv)


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
    model.load_state_dict(torch.load('gpt.pkl', map_location='cpu'))
    model = model.cuda(args.devices[0])
    model.eval()
    infer = Inferencer(args, model)
    infer.pred(args.text)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT Training')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_attention_heads', type=int, default=16)
    parser.add_argument('--vocab_size', type=int, default=119547)
    parser.add_argument('--dropout_prob', type=int, default=0.1)
    parser.add_argument('--max_position_embeddings', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--ignore_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=float, default=0)
    parser.add_argument('--devices', type=list, default=[0])
    parser.add_argument('--data', type=str, default="./minidata")

    parser.add_argument('--lr_base', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.001)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    parser.add_argument('--text', type=str, default='你好啊')

    args = parser.parse_args()
    main(args)
