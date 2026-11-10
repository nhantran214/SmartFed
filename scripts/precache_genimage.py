#!/usr/bin/env python3
"""Pre-populate the local SSD image cache for GenImage / DiffusionFace.

Usage (from repo root, smartfed env)::

    python scripts/precache_genimage.py
    python scripts/precache_genimage.py --clients adm BigGAN  # subset
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from Dataset.init_dataset import genimage
from Dataset.image_cache import prewarm_paths


def main():
    parser = argparse.ArgumentParser(description='Prewarm SmartFed SSD image cache')
    parser.add_argument(
        '--clients', nargs='*', default=None,
        help='Client folder names (default: all in Config.DATASET_LIST + public)',
    )
    parser.add_argument('--skip-public', action='store_true')
    args = parser.parse_args()

    Config.DATA_CACHE_ENABLED = True
    os.makedirs(Config.DATA_CACHE_DIR, exist_ok=True)
    print(f'Cache dir: {Config.DATA_CACHE_DIR}')
    print(f'Image size: {Config.DATA_CACHE_IMAGE_SIZE}')

    names = args.clients or list(Config.DATASET_LIST)
    roots = [os.path.join(Config.PRIVATE_DATASET_DIR, n) for n in names]
    if not args.skip_public and getattr(Config, 'PUBLIC_DATASET_DIR', None):
        roots.append(Config.PUBLIC_DATASET_DIR)

    total_ok = total_fail = 0
    for root in roots:
        if not os.path.isdir(root):
            print(f'Skip missing: {root}')
            continue
        for train_flag in (True, False):
            split = 'train' if train_flag else 'val'
            print(f'=== {root} [{split}] ===')
            ds = genimage(root=root, train=train_flag, transform=None)
            ok, fail = prewarm_paths(ds.data)
            print(f'  done: ok={ok}, fail={fail}')
            total_ok += ok
            total_fail += fail

    print(f'All done. ok={total_ok}, fail={total_fail}')
    print(f'Cache root: {Config.DATA_CACHE_DIR}')


if __name__ == '__main__':
    main()
