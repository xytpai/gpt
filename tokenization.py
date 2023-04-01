import os
import argparse
import tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

VOCAB_FILE = 'vocab.json'


class Tokenizer(object):
    def __init__(self):
        if os.path.exists(VOCAB_FILE):
            print('load vocab file')
            self.tokenizer = tokenizers.Tokenizer.from_file(VOCAB_FILE)
        else:
            self.tokenizer = tokenizers.Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()

    def text_to_ids(self, text):
        return self.tokenizer.encode(text).ids

    def ids_to_text(self, ids):
        return self.tokenizer.decode(ids)

    def train(self, text_files):
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=119547)
        self.tokenizer.train(files=text_files, trainer=trainer)

    def memory_efficient_train(self, text_files, chunk):
        f_c = 0
        flist_ = []
        flist = []
        for file in text_files:
            flist_.append(file)
            f_c += 1
            if f_c >= chunk:
                flist.append(flist_)
                flist_ = []
                f_c = 0
        if len(flist_) > 0:
            flist.append(flist_)
        for fs in flist:
            print('Processing ' + str(fs))
            self.train(fs)
            self.save()

    def save(self):
        self.tokenizer.save(VOCAB_FILE)


def main(args):
    tokenizer = Tokenizer()
    files = []
    for path, dir_list, file_list in os.walk(args.dir):
        for file in file_list:
            if file.endswith('.txt'):
                files.append(os.path.join(path, file))
    tokenizer.memory_efficient_train(files, args.chunk)

    tokenizer2 = Tokenizer()
    ids = tokenizer2.text_to_ids('Hello world !')
    print(ids)
    ids_inv = tokenizer2.ids_to_text(ids)
    print(ids_inv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BPE Tokenizer Training')
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--chunk', type=int, default=100000)
    args = parser.parse_args()
    main(args)
