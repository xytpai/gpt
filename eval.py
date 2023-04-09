import argparse
from dacite import from_dict
import torch
import re

import modeling
import tokenization
from configs import gptconfigs
from train_level_zero import load_model


class Inferencer(object):
    def __init__(self, args, model):
        self.tokenizer = tokenization.Tokenizer()
        self.args = args
        self.model = model
        self.model.eval()

    def pred(self, text):
        bos0 = self.tokenizer.text_to_ids('[BOS0]')[0]
        eos = self.tokenizer.text_to_ids('[EOS]')[0]
        ids_ = self.tokenizer.text_to_ids(text)
        ids = [bos0] + ids_ + [eos]
        d = self.args.devices[0]
        ids = torch.LongTensor(ids).view(1, -1)
        if isinstance(d, int):
            ids = ids.cuda(d)
        with torch.no_grad():
            ids_o = self.model.generate(self.tokenizer, ids, self.args.max_position_embeddings, temperature=0.6)
            ids_o = list(ids_o.cpu()[0].numpy())
            ids_inv = self.tokenizer.ids_to_text(ids_o)
            text = re.sub(r'\s+([\u4e00-\u9fff]+)\s+', r'\1', ids_inv)
            # text = ids_inv.replace(' ', '')
            print(text)


def main(args):
    config = from_dict(data_class=modeling.GPTConfig, data=args)
    model = modeling.GPT(config)
    load_model(args, model)
    if isinstance(args.devices[0], int):
        model = model.cuda(args.devices[0])
    model.eval()
    infer = Inferencer(args, model)
    infer.pred(args.text)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT Evaluating')
    parser.add_argument('--model', type=str, default='nano')
    parser.add_argument('--devices', type=list, default=['cpu'])
    parser.add_argument('--text', type=str, default='')
    args = parser.parse_args()
    new_args = gptconfigs[args.model]
    new_args.update(vars(args))
    main(new_args)
