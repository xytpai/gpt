import os
import random
import time
import math
import argparse
from tqdm import tqdm
from dacite import from_dict
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

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

from train_level_zero import load_model, prepare_device, prepare_model, \
    prepare_loader, prepare_optimizer, prepare_lr_scheduler, Trainer, parse_args


def prepare_dataset(args):
    dataset = datasets.RelationGPTDataset(args)
    return dataset


class CriticTrainer(object):
    def step_epoch(self):
        if self.milestone >= self.args.end:
            return True
        for i, (x, y) in enumerate(self.loader):

            # accumulate gradient
            self.batch_size_count += x.shape[0]
            with autocast():
                output, loss = self.model(x, y)
            self.scaler.scale(loss / float(self.args.gradient_accumulation_steps)).backward()
            if self.grad_acc_count < self.args.gradient_accumulation_steps:
                self.grad_acc_count += 1
                continue
            batch_size = self.batch_size_count
            self.batch_size_count = 0
            self.grad_acc_count = 0

            # step optimizer
            if self.args.grad_clip > 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.lr_scheduler.step()

            # setp utils
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
    args.world_size = world_size
    prepare_device(args)
    model = prepare_model(args)
    dataset = prepare_dataset(args)
    loader =prepare_loader(args, dataset)
    opt = prepare_optimizer(args, model)
    lr_scheduler = prepare_lr_scheduler(args, opt)
    trainer = CriticTrainer(args, model, dataset, loader, opt, lr_scheduler)
    while True:
        if trainer.step_epoch():
            break
    destroy_process_group()
    if args.rank == 0:
        print('Training procedure finished!')


if __name__ == '__main__':
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(args, world_size), nprocs=world_size)
