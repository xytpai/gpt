import os
import random
import time
import argparse
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

# ddp
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp

import modeling
import tokenization
import datasets
from configs import gptconfigs

WEIGHT_DIR_NAME = './checkpoints'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'


def load_model(args, model):
    def get_milestone(fname):
        return int(fname.split('_')[1].replace('.ckpt', ''))
    for path, dir_list, file_list in os.walk(WEIGHT_DIR_NAME):
        files = []
        for file in file_list:
            if file.endswith('.ckpt') and file.startswith(args.model):
                files.append(file)
            files = sorted(files, key=lambda x : get_milestone(x), reverse=True)
            args.begin = get_milestone(files[0])
            ckpt = torch.load(os.path.join(path, files[0]), map_location='cpu')
            model.load_state_dict(ckpt['model'])
            args.ckpt = ckpt


def prepare_device(args):
    print('launch rank:' + str(args.rank))
    rank = args.rank
    torch.cuda.set_device(rank)
    seed = args.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    init_process_group(backend='nccl', rank=rank, world_size=args.world_size)


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
    if not args.no_load:
        load_model(args, model)
    model = model.cuda(args.rank)
    model = DDP(model, device_ids=[args.rank])
    model.train()
    return model


def prepare_dataset(args):
    dataset = datasets.GPTDataset(args)
    return dataset


def prepare_loader(args, dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, pin_memory=True,
        shuffle=False, collate_fn=dataset.collate_fn, sampler=DistributedSampler(dataset, shuffle=False))
    return loader


def prepare_optimizer(args, model):
    lr_base = args.lr_base
    opt = torch.optim.AdamW(model.parameters(), lr=lr_base, weight_decay=args.weight_decay)
    return opt


def prepare_lr_scheduler(args, opt):
    training_steps = float(args.end / args.batch_size)
    warmup_steps = 1 + training_steps * 0.002
    def lr_lambda(step):
        if step < warmup_steps:
            return float(max(1.0, step) / warmup_steps)
        else:
            return max(0.0, float(training_steps - step) \
                / float(max(1.0, training_steps - warmup_steps)))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    if args.get('ckpt', None) is not None:
        lr_scheduler.load_state_dict(args.ckpt['lr_scheduler'])
    return lr_scheduler


class Trainer(object):
    def __init__(self, args, model, dataset, loader, opt, lr_scheduler):
        self.args = args
        self.model_name = args.model
        self.rank = args.rank
        self.model = model
        self.dataset = dataset
        self.loader = loader
        self.opt = opt
        self.lr_scheduler = lr_scheduler
        self.__create_tools()
    
    def __del__(self):
        self.__delete_tools()

    def __create_tools(self):
        self.milestone = self.args.begin
        if self.rank == 0:
            self.pbar = tqdm(total=self.args.end)
            self.pbar.update(self.args.begin)
            self.tensorboard = SummaryWriter('summary')
            self.count_for_save = 0

    def __delete_tools(self):
        if self.rank == 0:
            self.tensorboard.close()
            self.pbar.close()

    def __step_tools(self, reduced_batch_size, reduced_loss):
        self.milestone += reduced_batch_size
        if self.rank == 0:
            cur_lr = float(self.lr_scheduler.get_last_lr()[0])
            self.count_for_save += reduced_batch_size
            maxmem = int(torch.cuda.max_memory_allocated(device=self.rank) / 1024 / 1024)
            info = 'loss:%f, maxMem:%dMB, lr:%f' % (reduced_loss, maxmem, cur_lr)
            self.pbar.set_description(info)
            self.tensorboard.add_scalar('train/loss', reduced_loss, self.milestone)
            self.tensorboard.add_scalar('train/lr', cur_lr, self.milestone)
            self.pbar.update(reduced_batch_size)

    def __save_weight(self):
        if not os.path.exists(WEIGHT_DIR_NAME):
            os.makedirs(WEIGHT_DIR_NAME)
        for path, dir_list, file_list in os.walk(WEIGHT_DIR_NAME):
            files = []
            for file in file_list:
                if file.endswith('.ckpt') and file.startswith(self.model_name):
                    files.append(os.path.join(path, file))
        files_rev = sorted(files, reverse=True)
        keep_files = files_rev[:self.args.num_save_files - 1]
        for file in files:
            if file not in keep_files:
                os.remove(file)
        weight_filename = os.path.join(WEIGHT_DIR_NAME, 
            self.model_name + '_' + str(self.milestone).zfill(10) + '.ckpt')
        model_sdict = self.model.module.state_dict()
        lr_scheduler_sdict = self.lr_scheduler.state_dict()
        torch.save({'model': model_sdict, 'lr_scheduler': lr_scheduler_sdict}, weight_filename)

    def __step_saver(self):
        if self.milestone >= self.args.end:
            if self.rank == 0:
                self.__save_weight()
            return True
        if self.rank == 0:
            if self.count_for_save >= self.args.save_interval:
                self.__save_weight()
                self.count_for_save = 0
        return False

    def step_epoch(self):
        if self.milestone >= self.args.end:
            return True
        for i, (x, y) in enumerate(self.loader):
            batch_size = x.shape[0]
            self.opt.zero_grad()
            output, loss = self.model(x, y)
            loss = loss.mean()
            loss.backward()
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.opt.step() # dist.sync
            self.lr_scheduler.step()
            bs_loss_t = torch.empty(2).double()
            bs_loss_t[0] = batch_size
            bs_loss_t[1] = loss.item()
            bs_loss_t = bs_loss_t.cuda(self.rank)
            all_reduce(bs_loss_t, op=ReduceOp.SUM)
            self.__step_tools(int(bs_loss_t[0].item()), float(bs_loss_t[1].item()) / self.args.world_size)
            if self.__step_saver():
                return True
        return False


def main(rank, args, world_size):
    args.rank = rank
    args.world_size =world_size
    prepare_device(args)
    model = prepare_model(args)
    dataset = prepare_dataset(args)
    loader =prepare_loader(args, dataset)
    opt = prepare_optimizer(args, model)
    lr_scheduler = prepare_lr_scheduler(args, opt)
    trainer = Trainer(args, model, dataset, loader, opt, lr_scheduler)
    while True:
        if trainer.step_epoch():
            break
    destroy_process_group()
    if args.rank == 0:
        print('Training procedure finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPT Training')
    parser.add_argument('--model', type=str, default='nano')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=float, default=0)
    parser.add_argument('--data', type=str, default="./minidata")
    parser.add_argument('--lr_base', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--no_load', action='store_true')
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int, default=14000)
    parser.add_argument('--save_interval', type=int, default=4000)
    parser.add_argument('--num_save_files', type=int, default=10)
    args = parser.parse_args()
    new_args = gptconfigs[args.model]
    new_args.update(vars(args))
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(new_args, world_size), nprocs=world_size)
