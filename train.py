import os
import random
import time
import argparse
import torch

import modeling
import tokenization
import datasets
from configs import gptconfigs


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
    # weight_decay = args.weight_decay
    # momentum = args.momentum
    # params = []
    # for key, value in model.named_parameters():
    #     if not value.requires_grad:
    #         continue
    #     _lr = lr_base
    #     _weight_decay = weight_decay
    #     if "bias" in key:
    #         _lr = lr_base * 2
    #         _weight_decay = 0
    #     params += [{"params": [value], "lr": _lr, "weight_decay": _weight_decay}]
    # opt = torch.optim.SGD(params, lr=_lr, momentum=momentum)
    opt = torch.optim.Adam(model.parameters(), lr=lr_base)
    return opt


class Trainer(object):
    def __init__(self, args, model, dataset, loader, opt):
        self.args = args
        self.model = model
        self.dataset = dataset
        self.loader = loader
        self.opt = opt
        self.step = 0
        self.epoch = 0
        self.trained_step = 0
        # lr
        # self.grad_clip = cfg['TRAIN']['GRAD_CLIP']
        # self.lr_base = cfg['TRAIN']['LR_BASE']
        # self.lr_gamma = cfg['TRAIN']['LR_GAMMA']
        # self.lr_schedule = cfg['TRAIN']['LR_SCHEDULE']
        # self.warmup_iters = cfg['TRAIN']['WARMUP_ITER']
        # self.warmup_factor = 1.0/3.0  
        # self.device = cfg['TRAIN']['DEVICES']
        # self.save = cfg['TRAIN']['SAVE']
        
    def step_epoch(self, save_last=False):
        for i, (x, y) in enumerate(self.loader):
            # lr function
            lr = self.args.lr_base
            self.trained_step += x.shape[0]
            # if self.step < self.warmup_iters:
            #     alpha = float(self.step) / self.warmup_iters
            #     warmup_factor = self.warmup_factor * (1.0 - alpha) + alpha
            #     lr = lr*warmup_factor 
            # else:
            #     for j in range(len(self.lr_schedule)):
            #         if self.step < self.lr_schedule[j]:
            #             break
            #         lr *= self.lr_gamma
            # for param_group in self.opt.param_groups:
            #     param_group['lr'] = lr
            # #########

            torch.cuda.synchronize()
            time_start = time.time()
            
            self.opt.zero_grad()
            output, loss = self.model(x, y)
            loss = loss.mean()
            loss.backward()
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.opt.step()

            maxmem = int(torch.cuda.max_memory_allocated(device=self.args.devices[0]) / 1024 / 1024)
            torch.cuda.synchronize()
            time_end = time.time()
            totaltime = int((time_end - time_start) * 1000)
            print('total_step:%d: epoch:%d, step:%d/%d, loss:%f, maxMem:%dMB, time:%dms, lr:%f' % \
                (self.step, self.epoch, self.trained_step, len(self.dataset), loss, maxmem, totaltime, lr))
            self.step += 1
        self.epoch += 1
        if self.trained_step >= args.end_step:
            torch.save(self.model.module.state_dict(), 'gpt.pkl')
            return True
        else:
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
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.001)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    parser.add_argument('--end_step', type=int, default=14322*2)

    args = parser.parse_args()
    new_args = gptconfigs[args.model]
    new_args.update(vars(args))
    main(new_args)
