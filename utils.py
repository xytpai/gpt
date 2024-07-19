# base
import os
import math
import random
from tqdm import tqdm
from dacite import from_dict

# pytorch core
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

# pytorch ddp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp

# ddp config
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'


class ModelLoader(object):
    '''
    [ Introduction ]
    This class is used for the unified management of loading and storing model checkpoints.

    [ Arguments ]
    - model: torch.nn model for loading.
    - checkpoints_dir: Folder for storing checkpoints.
        For example:
        ./checkpoints
            ${fname_prefix}_0000001.ckpt (0000001 is the training step)
            ${fname_prefix}_0000002.ckpt
    - fname_prefix: See above.
    - max_checkpoints: The maximum number of checkpoint files to retain.

    [ Methods ]
    - load
        Search for .ckpt files in the checkpoints_dir that meet the prefix requirements, 
        sort them by the filename suffix (denoting step), 
        and load the state dict of the file with the largest step into the model.        

        - to_half: Whether the model needs to be converted to half-precision before loading the checkpoint.
        - no_checkpoint: If set to True, the model will not load the checkpoint.
        - no_ddp: If set to True, the model will not use ddp.
        - to_cuda: If set to False, the model will not copy to cuda after loading the checkpoint.
        - rank: Specify the device id when to_cuda is set to True.
        - freeze_patterns: A list specifying the patterns for freezing parameters, such as: ['attention', 'mlp'].

    - store
        Store the current model parameters to a new checkpoint file, using the step as the filename suffix. 
        It may delete the file with the smallest step to ensure that there are at most max_checkpoints files.
        If checkpoints_dir does not exist, it will be created.

        - step: Mainly used for the filename suffix. (${fname_prefix}_${padded_step}.ckpt)
        - additional_dict: Additional state dict to be added.
    
    - model
        Return the self.model
    
    - ckpt
        Return the self.ckpt
    '''
    def __init__(self, 
                 model, 
                 checkpoints_dir: str, 
                 fname_prefix: str, 
                 max_checkpoints: int):
        self.model = model
        self.checkpoints_dir = checkpoints_dir
        self.fname_prefix = fname_prefix
        self.max_checkpoints = max_checkpoints
        self.ckpt = None
    
    def __freeze_layers(self, freeze_patterns: list):
        for pattern in freeze_patterns:
            for name, param in self.model.named_parameters():
                if pattern in name:
                    print('freeze ' + name)
                    param.requires_grad = False
    
    def __get_step_from_fname(self, fname):
        return int(fname.strip().split('_')[-1].replace('.ckpt', ''))
    
    def __load_checkpoint(self):
        files = []
        for path, _, file_list in os.walk(self.checkpoints_dir):
            for file in file_list:
                file = file.strip()
                if file.endswith('.ckpt') and file.startswith(self.fname_prefix):
                    files.append(file)
        files = sorted(files, key=lambda x : self.__get_step_from_fname(x), reverse=True)
        if len(files) > 0:
            latest_checkpoint_file = files[0]
            ckpt = torch.load(os.path.join(path, latest_checkpoint_file), map_location='cpu')
            ckpt_model = ckpt.get('model', None)
            if ckpt_model is None:
                missing_keys, unexpected_keys = self.model.load_state_dict(ckpt, strict=False)
                ckpt_model = ckpt
                ckpt = {}
                ckpt['model'] = ckpt_model
                ckpt['step'] = 0
            else:
                missing_keys, unexpected_keys = self.model.load_state_dict(ckpt_model, strict=False)
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                print('load model: ' + str({'missing_keys':missing_keys, 'unexpected_keys':unexpected_keys}))
        else:
            raise FileNotFoundError('No checkpoint found')
        self.ckpt = ckpt
    
    def load(self, 
             to_half: bool=False, 
             no_checkpoint: bool=False, 
             no_ddp: bool=False,
             to_cuda: bool=True,
             rank: int=0, 
             freeze_patterns: list=[]):
        self.to_half = to_half
        self.no_checkpoint = no_checkpoint
        self.no_ddp = no_ddp
        self.to_cuda = to_cuda
        self.rank = rank
        self.freeze_patterns = freeze_patterns
        if self.to_half:
            self.model = self.model.half()
        if not self.no_checkpoint:
            self.__load_checkpoint()
        self.__freeze_layers(self.freeze_patterns)
        if to_cuda:
            self.model = self.model.cuda(self.rank)
        if not no_ddp:
            self.model = DDP(self.model, device_ids=[self.rank])
        self.model.train()

    def store(self, step, additional_dict={}):
        checkpoints_dir = self.checkpoints_dir
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        for path, _, file_list in os.walk(checkpoints_dir):
            files = []
            for file in file_list:
                if file.endswith('.ckpt') and file.startswith(self.fname_prefix):
                    files.append(os.path.join(path, file))
        files_rev = sorted(files, reverse=True)
        keep_files = files_rev[:self.max_checkpoints - 1]
        for file in files:
            if file not in keep_files:
                os.remove(file)
        ckpt_filename = os.path.join(checkpoints_dir, 
            self.fname_prefix + '_' + str(step).zfill(10) + '.ckpt')
        if not self.no_ddp:
            model_sdict = self.model.module.state_dict()
        else:
            model_sdict = self.model.state_dict()
        state_dict = {
            'model': model_sdict,
            'step': step
        }
        state_dict.update(additional_dict)
        torch.save(state_dict, ckpt_filename)
    
    def model(self):
        return self.model
    
    def ckpt(self):
        return self.ckpt


class DDPContext(object):
    '''
    [ Introduction ]
    Initialize the current GPU and set the seed.
    '''
    def __init__(self, rank, world_size, seed, backend='nccl'):
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.backend = backend
    
    def __enter__(self):
        print('launch rank:' + str(self.rank))
        torch.cuda.set_device(self.rank)
        seed = self.seed + self.rank
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        init_process_group(backend=self.backend, rank=self.rank, world_size=self.world_size)
    
    def __exit__(self, exc_type, exc_value, traceback):
        destroy_process_group()


class TrainingScheduler(object):

    @staticmethod
    def prepare_dataloader(dataset, batch_size):
        # Always in ddp
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            pin_memory=True,
            shuffle=False, 
            collate_fn=dataset.collate_fn, 
            sampler=DistributedSampler(dataset, shuffle=False))
        return loader

    @staticmethod
    def prepare_optimizer(
        model, 
        lr_base, 
        weight_decay, 
        weight_decay_whitelist=(torch.nn.Linear, ), 
        weight_decay_blacklist=(torch.nn.Embedding, )):
        # AdamW optimizer
        decay = set()
        no_decay = set()
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                full_param_name = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(full_param_name)
                elif pn.endswith('weight') and isinstance(m, weight_decay_whitelist):
                    decay.add(full_param_name)
                elif pn.endswith('weight') and isinstance(m, weight_decay_blacklist):
                    no_decay.add(full_param_name)
        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(optim_groups, lr=lr_base, betas=(0.9, 0.95))
        return opt

    @staticmethod
    def prepare_lr_scheduler(end_step, batch_size, lr_base, lr_min, opt, ckpt):
        # LambdaLR
        training_steps = float(end_step / batch_size)
        warmup_steps = 2000
        lr_decay_steps = training_steps
        lr_min = float(lr_min)
        lr_base = float(lr_base)
        def lr_lambda(step):
            if step < warmup_steps:
                return float(max(1.0, step) / warmup_steps)
            elif step > lr_decay_steps:
                return lr_min / lr_base
            else:
                decay_ratio = (step - warmup_steps) / (lr_decay_steps - warmup_steps)
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
                return (lr_min + coeff * (lr_base - lr_min)) / lr_base
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        lr_sd = ckpt.get('lr_scheduler', None)
        if lr_sd is not None:
            lr_scheduler.load_state_dict(lr_sd)
        return lr_scheduler

    def __init__(self, 
                 model,
                 checkpoints_dir: str,
                 fname_prefix: str,
                 max_checkpoints: int,
                 freeze_patterns: list,
                 dataset,
                 batch_size: int,
                 lr_base: float,
                 weight_decay: float,
                 estimated_end_step: int,
                 lr_min: float,
                 save_interval: int,
                 gradient_accumulation_steps: int,
                 grad_clip: float=0.0,
                 weight_decay_blacklist: tuple=(torch.nn.Embedding, )):
        self.model = model
        self.checkpoints_dir = checkpoints_dir
        self.fname_prefix = fname_prefix
        self.max_checkpoints = max_checkpoints
        self.freeze_patterns = freeze_patterns
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr_base = lr_base
        self.weight_decay = weight_decay
        self.estimated_end_step = estimated_end_step
        self.lr_min = lr_min
        self.save_interval = save_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip = grad_clip
        self.weight_decay_blacklist = weight_decay_blacklist
    
    def initialize(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.modelloader = ModelLoader(self.model, self.checkpoints_dir, self.fname_prefix, self.max_checkpoints)
        self.modelloader.load(rank=self.rank, freeze_patterns=self.freeze_patterns)
        self.dataloader = self.prepare_dataloader(self.dataset, self.batch_size)
        self.opt = self.prepare_optimizer(self.modelloader.model, self.lr_base, self.weight_decay,
                                          weight_decay_blacklist=self.weight_decay_blacklist)
        self.lr_scheduler = self.prepare_lr_scheduler(
            self.estimated_end_step, self.batch_size, self.lr_base, self.lr_min, self.opt, self.modelloader.ckpt)
        self.step = int(self.modelloader.ckpt['step'])
        # grad acc
        self.grad_acc_count = 0
        # scaler
        self.scaler = GradScaler()
        # system
        self.__create_system()
    
    def __create_system(self):
        if self.rank == 0:
            self.pbar = tqdm(total=self.estimated_end_step, ncols=80)
            self.pbar.update(self.step)
            self.tensorboard = SummaryWriter('summary')
            self.count_for_save = 0

    def delete(self):
        if self.rank == 0:
            self.tensorboard.close()
            self.pbar.close()
        
    def __step_system(self, reduced_loss):
        is_end = False
        self.step += 1
        if self.rank == 0:
            cur_lr = float(self.lr_scheduler.get_last_lr()[0])
            self.count_for_save += 1
            info = 'loss[%f]' % (reduced_loss)
            self.pbar.set_description(info)
            self.tensorboard.add_scalar('train/loss', reduced_loss, self.step)
            self.tensorboard.add_scalar('train/lr', cur_lr, self.step)
            self.pbar.update(1)
        if self.step >= self.estimated_end_step:
            if self.rank == 0:
                lr_scheduler_sdict = self.lr_scheduler.state_dict()
                self.modelloader.store(self.step, {'lr_scheduler': lr_scheduler_sdict})
                self.__store_checkpoint()
            is_end = True
        if self.rank == 0:
            if self.count_for_save >= self.save_interval:
                lr_scheduler_sdict = self.lr_scheduler.state_dict()
                self.modelloader.store(self.step, {'lr_scheduler': lr_scheduler_sdict})
                self.__store_checkpoint()
                self.count_for_save = 0
        return is_end
        
    def step_epoch(self):
        if self.step >= self.estimated_end_step:
            return True
        for i, (x, y) in enumerate(self.dataloader):
            # accumulate gradient
            with autocast():
                state, loss = self.modelloader.model(x, y)
            self.scaler.scale(loss / float(self.gradient_accumulation_steps)).backward()
            if self.grad_acc_count < self.gradient_accumulation_steps - 1:
                self.grad_acc_count += 1
                continue
            self.grad_acc_count = 0
            # step optimizer
            if self.grad_clip > 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.modelloader.model.parameters(), self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.lr_scheduler.step()
            # step system
            loss_t = torch.full((1, ), loss.item(), dtype=torch.double).cuda(self.rank)
            all_reduce(loss_t, op=ReduceOp.SUM)
            reduced_loss = float(loss_t[0].item()) / self.args.world_size
            is_end = self.__step_system(reduced_loss)
            if is_end:
                return True
        return False
