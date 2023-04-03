import argparse
from dacite import from_dict
import torch

import modeling
import tokenization
from configs import gptconfigs
from train import load_model


class Inferencer(object):
    def __init__(self, args, model):
        self.tokenizer = tokenization.Tokenizer()
        self.args = args
        self.model = model
        self.model.eval()

    def pred(self, text):
        ids = self.tokenizer.text_to_ids(text)
        ids = torch.LongTensor(ids).view(1, -1).cuda()
        with torch.no_grad():
            ids_o = self.model.generate(ids, self.args.max_position_embeddings)
            ids_o = list(ids_o.cpu()[0].numpy())
            ids_inv = self.tokenizer.ids_to_text(ids_o)
            print(ids_inv)


def main(args):
    config = from_dict(data_class=modeling.GPTConfig, data=args)
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
