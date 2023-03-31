import os
import argparse
import json


def main(args):
    assert args.input.endswith('.txt') or args.input.endswith('.data')
    with open(args.input, 'r', encoding='utf-8') as f:
        txt = f.read()
    datas = []
    target_length = args.len
    while len(txt) > 0:
        txt0 = txt[:target_length]
        txt1 = txt[target_length//2:target_length//2+target_length]
        if len(txt0) >= 10:
            datas.append(txt0)
        if len(txt1) >= 10:
            datas.append(txt1)
        txt = txt[target_length:]
    outfile = args.input.replace('.txt', '.jsonl').replace('.data', '.jsonl')
    with open(outfile, 'w', encoding='utf-8') as f:
        for data in datas:
            f.write(json.dumps({'data': data, 'length': len(data)}))
            f.write('\n')
    with open(outfile, 'r',  encoding='utf-8') as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
    print('done, examples:', lines[:2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate .jsonl from any text')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--len', type=int, required=True)
    args = parser.parse_args()
    main(args)
