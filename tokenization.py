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
            # print('load vocab file')
            self.tokenizer = tokenizers.Tokenizer.from_file(VOCAB_FILE)
        else:
            self.tokenizer = tokenizers.Tokenizer(BPE())
        self.tokenizer.pre_tokenizer = Whitespace()

    def text_to_ids(self, text):
        return self.tokenizer.encode(text).ids

    def ids_to_text(self, ids):
        return self.tokenizer.decode(ids)

    def train(self, text_files):
        trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=13088)
        self.tokenizer.train(files=text_files, trainer=trainer)

    def save(self):
        self.tokenizer.save(VOCAB_FILE)


def main(args):
    tokenizer = Tokenizer()
    files = []
    for path, dir_list, file_list in os.walk(args.dir):
        for file in file_list:
            if file.endswith('.json'):
                files.append(os.path.join(path, file))
    print(files)
    tokenizer.train(files)
    tokenizer.save()

    tokenizer2 = Tokenizer()
    ids = tokenizer2.text_to_ids('你好啊![SEP]')
    print(ids)
    print(tokenizer2.ids_to_text(ids))
    ids = tokenizer2.text_to_ids('你 好 啊![SEP]')
    print(ids)
    print(tokenizer2.ids_to_text(ids))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BPE Tokenizer Training')
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
