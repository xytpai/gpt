import os
import random
import json
import torch
import tokenization
from torch.utils.data import Dataset, DataLoader


def get_data_file_list(filedir):
    data_file_list = []
    for path, dir_list, file_list in os.walk(filedir):
        for file in file_list:
            if file.endswith('.txtl'):
                data_file_list.append(os.path.join(path, file))
    random.shuffle(data_file_list)
    return data_file_list


class GPTDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = tokenization.Tokenizer()
        self.eos = self.tokenizer.text_to_ids('[EOS]')[0]
        data_file_list = get_data_file_list(args.data)
        data_info_list = []
        total_lines = 0
        CACHE_FILE = '.datacache'
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache = json.loads(f.read())
            for file in data_file_list:
                print('init ' + file + ' from .datacache')
                num_lines = cache[file]
                prefix = total_lines
                total_lines += num_lines
                data_info_list.append({'filename':file, 'start':prefix, 'end':total_lines})
        else:
            cache = {}
            for file in data_file_list:
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
        # print('data_info_list:', data_info_list)

        self.set_current_file(self.data_info_list[0]['filename'])
        self.max_position_embeddings = args.max_position_embeddings
        self.ignore_index = args.ignore_index

    def set_current_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
            self.current_file = lines
        self.current_filename = filename

    def __len__(self):
        return self.total_lines

    def get_line_ids(self, index):
        idx = 0
        for i, info in enumerate(self.data_info_list):
            if index < info['end']:
                idx = i
                break
        if self.current_filename != self.data_info_list[idx]['filename']:
            self.set_current_file(self.data_info_list[idx]['filename'])
        line = self.current_file[index - self.data_info_list[idx]['start']]
        ids = self.tokenizer.text_to_ids(line)
        ids = ids[:self.max_position_embeddings]
        return ids

    def __getitem__(self, index):
        ids = self.get_line_ids(index)
        return torch.LongTensor(ids), torch.LongTensor(ids[1:])

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
            index = torch.argmax(torch.eq(ys[b], self.eos).long()) + 1
            out_y[b, :index] = self.ignore_index
        return out_x, out_y


if __name__ == '__main__':
    from configs import gptconfig_nano
    gptconfig_nano.data = './minidata'
    gptconfig_nano.batch_size = 2
    dataset = GPTDataset(gptconfig_nano)
    loader = DataLoader(dataset, batch_size=gptconfig_nano.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)
    for data in loader:
        x, y = data
        print(x.shape, x.dtype)
        print(x)
        print(y)
        raise
