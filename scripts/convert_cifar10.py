#!/usr/bin/env python3
"""Convert official CIFAR-10 pickles to SmartFed image folders.

Default pickle root (reviewer path):
  /media/ethan/N-hub/mcsp_ubuntu/mcsp/FedAIDET/data/cifar-10-python
Output (loaders expect train/<class>/ and test/<class>/):
  <repo>/data/cifar
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
from PIL import Image

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


def unpickle(path: str):
    with open(path, 'rb') as fh:
        return pickle.load(fh, encoding='bytes')


def _find_batches_dir(pickle_root: str) -> str:
    """Accept either .../cifar-10-python or .../cifar-10-python/cifar-10-batches-py."""
    direct = os.path.join(pickle_root, 'data_batch_1')
    if os.path.isfile(direct):
        return pickle_root
    nested = os.path.join(pickle_root, 'cifar-10-batches-py')
    if os.path.isfile(os.path.join(nested, 'data_batch_1')):
        return nested
    raise FileNotFoundError(
        f'Cannot find data_batch_1 under {pickle_root!r} or {nested!r}'
    )


def convert(pickle_root: str, output_dir: str) -> None:
    batches_dir = _find_batches_dir(pickle_root)
    for split in ('train', 'test'):
        for name in CLASS_NAMES:
            os.makedirs(os.path.join(output_dir, split, name), exist_ok=True)

    print(f'Reading pickles from {batches_dir}')
    print(f'Writing images to {output_dir}')

    # Training batches 1–5
    for batch_num in range(1, 6):
        batch = unpickle(os.path.join(batches_dir, f'data_batch_{batch_num}'))
        data = batch[b'data']
        labels = batch[b'labels']
        for i, (row, label) in enumerate(zip(data, labels)):
            img = row.reshape(3, 32, 32).transpose(1, 2, 0)
            path = os.path.join(
                output_dir, 'train', CLASS_NAMES[label],
                f'batch{batch_num}_{i:05d}.png',
            )
            Image.fromarray(img.astype(np.uint8)).save(path)
        print(f'  wrote data_batch_{batch_num} ({len(labels)} images)')

    test = unpickle(os.path.join(batches_dir, 'test_batch'))
    data = test[b'data']
    labels = test[b'labels']
    for i, (row, label) in enumerate(zip(data, labels)):
        img = row.reshape(3, 32, 32).transpose(1, 2, 0)
        path = os.path.join(
            output_dir, 'test', CLASS_NAMES[label], f'test_{i:05d}.png',
        )
        Image.fromarray(img.astype(np.uint8)).save(path)
    print(f'  wrote test_batch ({len(labels)} images)')
    print('Done.')


def main(argv=None):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--pickle-root',
        default=os.environ.get(
            'SMARTFED_CIFAR10_PICKLE_DIR',
            '/media/ethan/N-hub/mcsp_ubuntu/mcsp/FedAIDET/data/cifar-10-python',
        ),
        help='Directory containing cifar-10-batches-py (or the batches themselves)',
    )
    parser.add_argument(
        '--output-dir',
        default=os.environ.get(
            'SMARTFED_PRIVATE_DATASET_DIR',
            os.path.join(repo, 'data', 'cifar'),
        ),
        help='Output image root with train/ and test/',
    )
    args = parser.parse_args(argv)
    convert(args.pickle_root, args.output_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
