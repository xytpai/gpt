import argparse
from dacite import from_dict
import torch
from xmlrpc.server import SimpleXMLRPCServer

import modeling
import tokenization
from configs import gptconfigs
from train_sft import load_model


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
            self.model = self.model.cuda(d)
        with torch.no_grad():
            ids_o = self.model.generate(self.tokenizer, ids, self.args.max_position_embeddings, temperature=0.5)
            return ids_o


def main(args):

    config = from_dict(data_class=modeling.GPTConfig, data=args)
    model = modeling.GPT(config)
    load_model(args, model)
    if isinstance(args.devices[0], int):
        model = model.cuda(args.devices[0])
    model.eval()
    infer = Inferencer(args, model)

    def response(str):
        print('user:'+str)
        return infer.pred(str)[1]

    server = SimpleXMLRPCServer(('192.168.50.19', 8888))
    server.register_function(response, 'get')
    server.serve_forever()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT Evaluating')
    parser.add_argument('--model', type=str, default='nano')
    parser.add_argument('--devices', type=list, default=['cpu'])
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    if args.cuda:
        args.devices = [0]
    new_args = gptconfigs[args.model]
    new_args.update(vars(args))
    main(new_args)
