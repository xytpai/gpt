import os
import random
import time
import argparse
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

import modeling
import tokenization
import datasets
from configs import gptconfigs


def load_model(args, model):
    def get_milestone(fname):
        return int(fname.split('_')[1].replace('.pkl', ''))
    for path, dir_list, file_list in os.walk('weights'):
        files = []
        for file in file_list:
            if file.endswith('.pkl') and file.startswith(args.model):
                files.append(file)
            files = sorted(files, key=lambda x : get_milestone(x), reverse=True)
            args.begin = get_milestone(files[0])
            model.load_state_dict(torch.load(os.path.join(path, files[0]), map_location='cpu'))


def prepare_device(args):
    torch.cuda.set_device(args.devices[0])
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_model(args):
    config = modeling.GPTConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        vocab_size=args.vocab_size,
        dropout_prob=args.dropout_prob,
        max_position_embeddings=args.max_position_embeddings,
        num_layers=args.num_layers,
        ignore_index=args.ignore_index,
    )
    print('config:', config)
    model = modeling.GPT(config)
    if args.load:
        load_model(args, model)
    model = torch.nn.DataParallel(model, device_ids=args.devices)
    model = model.cuda(args.devices[0])
    model.train()
    return model


def prepare_dataset(args):
    dataset = datasets.GPTDataset(args)
    return dataset


def prepare_loader(args, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate_fn)
    return loader


def prepare_optimizer(args, model):
    lr_base = args.lr_base
    opt = torch.optim.Adam(model.parameters(), lr=lr_base)
    return opt


class Trainer(object):
    def __init__(self, args, model, dataset, loader, opt):
        self.args = args
        self.model_name = args.model
        self.model = model
        self.dataset = dataset
        self.loader = loader
        self.opt = opt
        self.milestone = args.begin
        self.pbar = tqdm(total=args.end)
        self.pbar.update(args.begin)
        self.tensorboard = SummaryWriter('summary')
        self.first_run = True
    
    def __del__(self):
        self.tensorboard.close()

    def get_weight_filename(self):
        dirname = './weights'
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        return os.path.join(dirname, self.model_name + '_' + str(self.milestone).zfill(10) + '.pkl')

    def step_epoch(self, save_last=False):
        if self.milestone >= self.args.end:
            self.pbar.close()
            return True
        for i, (x, y) in enumerate(self.loader):
            batch_size = x.shape[0]
            lr = self.args.lr_base
            torch.cuda.synchronize()
            time_start = time.time()
            self.opt.zero_grad()
            if self.first_run:
                # self.tensorboard.add_graph(self.model, [x, y])
                self.first_run = False
            output, loss = self.model(x, y)
            loss = loss.mean()
            loss.backward()
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.opt.step()
            torch.cuda.synchronize()
            time_end = time.time()
            totaltime = int((time_end - time_start) * 1000)
            maxmem = int(torch.cuda.max_memory_allocated(device=self.args.devices[0]) / 1024 / 1024)
            self.milestone += batch_size
            info = 'loss:%f, maxMem:%dMB, time:%dms, lr:%f' % (loss, maxmem, totaltime, lr)
            self.pbar.set_description(info)
            self.tensorboard.add_scalar('loss', loss, self.milestone)
            self.pbar.update(batch_size)
            if self.milestone >= self.args.end:
                torch.save(self.model.module.state_dict(), self.get_weight_filename())
                self.pbar.close()
                return True
            elif self.milestone % self.args.save_interval == 0:
                torch.save(self.model.module.state_dict(), self.get_weight_filename())
        return False


def main(args):
    prepare_device(args)
    model = prepare_model(args)
    dataset = prepare_dataset(args)
    loader =prepare_loader(args, dataset)
    opt = prepare_optimizer(args, model)
    trainer = Trainer(args, model, dataset, loader, opt)
    while True:
        if trainer.step_epoch():
            break
    print('Training procedure finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT Training')
    parser.add_argument('--model', type=str, default='nano')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=float, default=0)
    parser.add_argument('--devices', type=list, default=[0])
    parser.add_argument('--data', type=str, default="./minidata")
    parser.add_argument('--lr_base', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--load', action='store_false')
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int, default=14000)
    parser.add_argument('--save_interval', type=int, default=1000)
    args = parser.parse_args()
    assert args.begin % args.save_interval == 0
    assert args.end % args.save_interval == 0
    new_args = gptconfigs[args.model]
    new_args.update(vars(args))
    main(new_args)
