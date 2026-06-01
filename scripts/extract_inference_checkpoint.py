#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract inference-only model weights from a training checkpoint.')
    parser.add_argument('input_checkpoint', type=Path)
    parser.add_argument('output_checkpoint', type=Path)
    parser.add_argument('--print-summary', action='store_true')
    args = parser.parse_args()

    ckpt = torch.load(args.input_checkpoint, map_location='cpu', mmap=True)

    if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
        state_dict = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], dict):
        state_dict = ckpt['state_dict']
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise TypeError(f'Unsupported checkpoint type: {type(ckpt).__name__}')

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.output_checkpoint)

    if args.print_summary:
        prefixes = Counter(k.split('.')[0] for k in state_dict.keys())
        print('num_model_keys', len(state_dict))
        print('top_prefix_counts', prefixes.most_common(20))

    print(f'Saved inference-only checkpoint to: {args.output_checkpoint}')


if __name__ == '__main__':
    main()
