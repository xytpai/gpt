import os
import random
import torch
import tokenization
from torch.utils.data import Dataset, DataLoader


def get_data_file_list(filedir):
    data_file_list = []
    for path, dir_list, file_list in os.walk(filedir):
        for file in file_list:
            if file.endswith('.data'):
                data_file_list.append(os.path.join(path, file))
    random.shuffle(data_file_list)
    return data_file_list


def get_filtered_lines(lines):
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if len(l) > 1]
    return lines


class GPTDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = tokenization.FullTokenizer('vocab.txt')
        data_file_list = get_data_file_list(args.data)
        data_info_list = []
        total_lines = 0
        for file in data_file_list:
            with open(file, 'r') as f:
                lines = get_filtered_lines(f.readlines())
                num_lines = len(lines)
                prefix = total_lines
                total_lines += num_lines
                data_info_list.append({'filename':file, 'start':prefix, 'end':total_lines})
        
        self.total_lines = total_lines
        self.data_info_list = data_info_list
        print('data_info_list:', data_info_list)

        self.set_current_file(self.data_info_list[0]['filename'])
        self.max_position_embeddings = args.max_position_embeddings
        self.ignore_index = args.ignore_index

    def set_current_file(self, filename):
        with open(filename, 'r') as f:
            self.current_file = get_filtered_lines(f.readlines())
        self.current_filename = filename
    
    def __len__(self):
        return self.total_lines
    
    def __getitem__(self, index):
        idx = 0
        for i, info in enumerate(self.data_info_list):
            if index < info['end']:
                idx = i
                break
        if self.current_filename != self.data_info_list[idx]['filename']:
            self.set_current_file(self.data_info_list[idx]['filename'])
        
        line = self.current_file[index - self.data_info_list[idx]['start']]

        tokens = self.tokenizer.tokenize(line)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = ids[:self.max_position_embeddings]
        return torch.LongTensor(ids), torch.LongTensor(ids[1:])

    def collate_fn(self, data):
        xs, ys = zip(*data)
        batch_size = len(xs)
        max_n = 0
        for b in range(batch_size):
            if xs[b].shape[0] > max_n: max_n = xs[b].shape[0]
        out_x = torch.full((batch_size, max_n), self.ignore_index)
        out_y = torch.full((batch_size, max_n), self.ignore_index)
        for b in range(batch_size):
            out_x[b, :xs[b].shape[0]] = xs[b]
            out_y[b, :ys[b].shape[0]] = ys[b]
        return out_x, out_y


if __name__ == '__main__':
    from dataclasses import dataclass
    @dataclass
    class DefaultConfig:
        hidden_size: int = 256
        num_attention_heads: int = 16
        vocab_size: int = 30522
        dropout_prob: float = 0.1
        max_position_embeddings: int = 512
        num_layers: int = 12
        ignore_index: int = -1
        data: str = './minidata'
        batch_size: int = 24
    
    args = DefaultConfig()
    dataset = GPTDataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)
    for data in loader:
        x, y = data
        print(x.shape, x.dtype)
        print(x)
        print(y)
        raise
