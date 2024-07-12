import os
import sys
import random
import json
import torch
import tokenization
from torch.utils.data import Dataset, DataLoader


def get_file_list(filedir, endswith='.jsonl', shuffle=True, sort=False):
    file_list_out = []
    for path, dir_list, file_list in os.walk(filedir):
        for file in file_list:
            if file.endswith(endswith):
                file_list_out.append(os.path.join(path, file))
    if shuffle:
        random.shuffle(file_list_out)
    if sort:
        file_list_out = sorted(file_list_out)
    return file_list_out


class JsonlinesFileListData(object):
    def __init__(self, filedir):
        super().__init__()
        file_list = get_file_list(filedir, '.jsonl', shuffle=False, sort=True)
        data_info_list = []
        total_lines = 0
        CACHE_FILE = '.datacache'
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache = json.loads(f.read())
            for file in file_list:
                print('init ' + file + ' from .datacache')
                num_lines = cache[file]
                prefix = total_lines
                total_lines += num_lines
                data_info_list.append({'filename':file, 'start':prefix, 'end':total_lines})
        else:
            cache = {}
            for file in file_list:
                with open(file, 'r') as f:
                    print('init ' + file)
                    num_lines = sum(1 for line in f)
                    cache[file] = num_lines
                    prefix = total_lines
                    total_lines += num_lines
                    data_info_list.append({'filename':file, 'start':prefix, 'end':total_lines})
            cache = json.dumps(cache)
            with open(CACHE_FILE, 'w') as f:
                f.write(cache)
        self.total_lines = total_lines
        self.data_info_list = data_info_list
        print('total_lines:', total_lines)
        print('data_info_list:', data_info_list)
        self.set_current_file(self.data_info_list[0]['filename'])
    
    def set_current_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            # random.shuffle(lines)
            self.current_file = lines
        self.current_filename = filename
    
    def __len__(self):
        return self.total_lines
    
    def get_line(self, global_index):
        idx = 0
        for i, info in enumerate(self.data_info_list):
            if global_index < info['end']:
                idx = i
                break
        if self.current_filename != self.data_info_list[idx]['filename']:
            self.set_current_file(self.data_info_list[idx]['filename'])
        line = self.current_file[global_index - self.data_info_list[idx]['start']]
        return line
    
    def __getitem__(self, index):
        return self.get_line(index)


class PromptDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = tokenization.SPTokenizer('tokenizer.model')
        self.data = JsonlinesFileListData(args.data)
        self.max_seq_length = args.max_seq_length
        self.ignore_index = args.ignore_index
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ids_x = []
        ids_y = []
        for item in json.loads(self.data[index]):
            instruction_string = "{}:".format(item['instruction'].strip())
            prompt_string = "{}\n".format(item['prompt'].strip())
            instruction_ids = self.tokenizer.encode(instruction_string, bos=False, eos=False)
            instruction_ignore_ids = [self.ignore_index for _ in range(len(instruction_ids))]
            prompt_ids = self.tokenizer.encode(prompt_string, bos=False, eos=False)
            ids_x += instruction_ids + [self.bos_id] + prompt_ids + [self.eos_id]
            ids_y += instruction_ignore_ids + [self.bos_id] + prompt_ids + [self.eos_id]
        ids_y.append(self.ignore_index)
        ids_x = torch.LongTensor(ids_x)
        ids_y = torch.LongTensor(ids_y[1:])
        return ids_x[:self.max_seq_length], ids_y[:self.max_seq_length]

    def collate_fn(self, data):
        xs, ys = zip(*data)
        batch_size = len(xs)
        max_n = 0
        for b in range(batch_size):
            if xs[b].shape[0] > max_n: max_n = xs[b].shape[0]
        out_x = torch.full((batch_size, max_n), 0).long()
        out_y = torch.full((batch_size, max_n), self.ignore_index).long()
        for b in range(batch_size):
            out_x[b, :xs[b].shape[0]] = xs[b]
            out_y[b, :ys[b].shape[0]] = ys[b]
            # find the first occurrence of eos_id and mask the previous prompt
            index = torch.argmax(torch.eq(ys[b], self.eos_id).long()) + 1
            out_y[b, :index] = self.ignore_index
        return out_x, out_y


if __name__ == '__main__':
    from configs import get_gpt_config
    config = get_gpt_config('6b')
    config.data = '/data/dataset/'
    dataset = PromptDataset(config)
    x, y = dataset[52]
    x = dataset.tokenizer.decode(x.tolist())
    y = dataset.tokenizer.decode(y.tolist())
    print(f"x:\n{x}")
    print(f"y:\n{y}")
