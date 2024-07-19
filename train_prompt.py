import torch
import argparse
import torch.multiprocessing as mp

from configs import get_gpt_config
from modeling import GPTXModel, RMSNormLayer
from datasets import PromptDataset
from utils import TrainingScheduler, DDPContext


def parse_args():
    parser = argparse.ArgumentParser(description='GPT Training')
    parser.add_argument('--level', type=str, default='6b')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=float, default=0)
    parser.add_argument('--data', type=str, default="/data/dataset/")
    parser.add_argument('--checkpoints_dir', type=str, default="./checkpoints")
    parser.add_argument('--lr_base', type=float, default=8e-6)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=2e-3)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--no_load', action='store_true')
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int, default=5000000)
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--num_save_files', type=int, default=10)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--freeze_patterns', type=str, default='[]')
    args = parser.parse_args()
    new_args = get_gpt_config(args.level)
    new_args.update(vars(args))
    new_args.freeze_patterns = eval(new_args.freeze_patterns)
    return new_args


def get_scheduler(args):
    model = GPTXModel(args)
    fname_prefix = 'gptx_' + args.level
    dataset = PromptDataset(args)
    training_scheduler = TrainingScheduler(
        model,
        args.checkpoints_dir,
        fname_prefix,
        args.num_save_files,
        args.freeze_patterns,
        dataset,
        args.batch_size,
        args.lr_base,
        args.weight_decay,
        args.end,
        args.lr_min,
        args.save_interval,
        args.gradient_accumulation_steps,
        args.grad_clip,
        weight_decay_blacklist=(torch.nn.Embedding, torch.nn, RMSNormLayer)
        )
    return training_scheduler


def main(rank, args, world_size):
    with DDPContext(rank, world_size, args.seed, 'nccl'):
        training_scheduler = get_scheduler(args)
        training_scheduler.initialize(rank, world_size)
        while True:
            if training_scheduler.step_epoch():
                break
        if rank == 0:
            print('Training procedure finished!')


if __name__ == '__main__':
    args = parse_args()
    print(args)
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(args, world_size), nprocs=world_size)
