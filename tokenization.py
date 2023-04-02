import os
import argparse
import tiktoken


class Tokenizer(object):
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding('gpt2')

    def text_to_ids(self, text):
        return self.tokenizer.encode(text)

    def ids_to_text(self, ids):
        return self.tokenizer.decode(ids)


if __name__ == '__main__':
    tokenizer = Tokenizer()
    ids = tokenizer.text_to_ids('Hello world!\n')
    ids_inv = tokenizer.ids_to_text(ids)
    print(ids)
    print(ids_inv)
