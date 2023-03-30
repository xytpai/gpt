import os
import argparse
import json


def main(args):
    assert args.input.endswith('.txt') or args.input.endswith('.data')
    with open(args.input, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
        lines = [l for l in lines if len(l) > 1]
    outfile = args.input.replace('.txt', '.jsonl').replace('.data', '.jsonl')
    with open(outfile, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(json.dumps({'data': line, 'length': len(line)}))
            f.write('\n')
    with open(outfile, 'r',  encoding='utf-8') as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
    print('done, examples:', lines[:2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate .jsonl from textlines')
    parser.add_argument('--input', type=str, default='')
    args = parser.parse_args()
    main(args)
