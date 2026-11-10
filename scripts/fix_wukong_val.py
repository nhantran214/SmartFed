#!/usr/bin/env python3
"""Fix / verify GenImage wukong/val (must not be a copy of vqdm/val).

Bug: wukong/val/{nature,ai} were identical to vqdm/val (fake images named VQDM_*).
Train/ai is correct wukong fakes. This script:

  1. Detects the duplication
  2. Backs up the bad val/ai folder
  3. Builds a new val/ai by holding out images from wukong/train/ai (no leakage)
  4. Leaves val/nature as ImageNet-val (shared reals across generators are OK)
  5. Purges SSD image-cache entries for replaced paths

After running, re-pretrain party 8 (wukong) so training never saw the new val images::

    PRETRAIN_PARTY_IDS = [8]
    python pretrain.py
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

# Allow `python scripts/fix_wukong_val.py` from repo root.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _list_files(d: Path):
    return sorted(p for p in d.iterdir() if p.is_file())


def detect(root: Path) -> dict:
    w_ai = set(p.name for p in _list_files(root / 'wukong' / 'val' / 'ai'))
    v_ai = set(p.name for p in _list_files(root / 'vqdm' / 'val' / 'ai'))
    w_nat = set(p.name for p in _list_files(root / 'wukong' / 'val' / 'nature'))
    v_nat = set(p.name for p in _list_files(root / 'vqdm' / 'val' / 'nature'))
    vqdm_named = sum(1 for n in w_ai if 'vqdm' in n.lower() or n.upper().startswith('VQDM'))
    wukong_train_named = sum(
        1 for n in (p.name for p in _list_files(root / 'wukong' / 'train' / 'ai'))
        if 'wukong' in n.lower()
    )
    return {
        'val_ai_identical_names': w_ai == v_ai and len(w_ai) > 0,
        'val_nature_identical_names': w_nat == v_nat and len(w_nat) > 0,
        'wukong_val_ai_vqdm_named': vqdm_named,
        'wukong_val_ai_count': len(w_ai),
        'wukong_train_ai_wukong_named': wukong_train_named,
        'intersect_ai': len(w_ai & v_ai),
    }


def purge_cache(paths):
    try:
        from Dataset.image_cache import cache_path_for, _cache_dir, _cache_enabled
    except Exception as e:
        print(f'[warn] cannot import image_cache ({e}); skip cache purge')
        return 0
    if not _cache_enabled():
        print('[info] DATA_CACHE_ENABLED=False; skip cache purge')
        return 0
    cache_dir = _cache_dir()
    if not cache_dir:
        return 0
    removed = 0
    for src in paths:
        cpath = cache_path_for(str(src), cache_dir)
        if os.path.isfile(cpath):
            try:
                os.remove(cpath)
                removed += 1
            except OSError:
                pass
    print(f'[info] purged {removed} SSD cache entries')
    return removed


def fix(root: Path, val_size: int, seed: int, dry_run: bool) -> None:
    info = detect(root)
    print('[detect]', info)
    if not info['val_ai_identical_names'] and info['wukong_val_ai_vqdm_named'] == 0:
        print('[ok] wukong/val/ai does not look like a vqdm duplicate; nothing to do')
        return

    train_ai = root / 'wukong' / 'train' / 'ai'
    val_ai = root / 'wukong' / 'val' / 'ai'
    backup = root / 'wukong' / 'val_ai_BAD_vqdm_duplicate'
    files = _list_files(train_ai)
    if len(files) < val_size:
        raise SystemExit(
            f'Need ≥{val_size} wukong/train/ai images to build val; found {len(files)}'
        )

    rng = random.Random(seed)
    holdout = sorted(rng.sample(files, val_size), key=lambda p: p.name)
    print(
        f'[plan] backup {val_ai} → {backup}; '
        f'move {val_size}/{len(files)} train/ai → new val/ai (seed={seed})'
    )
    if dry_run:
        print('[dry-run] no changes written')
        return

    if backup.exists():
        print(f'[info] backup already exists: {backup}')
    else:
        print(f'[info] moving bad val/ai → {backup}')
        shutil.move(str(val_ai), str(backup))

    val_ai.mkdir(parents=True, exist_ok=True)
    old_paths = list(_list_files(backup)) if backup.exists() else []
    new_paths = []
    for src in holdout:
        dst = val_ai / src.name
        shutil.move(str(src), str(dst))
        new_paths.append(dst)

    print(
        f'[done] new val/ai={len(_list_files(val_ai))}, '
        f'remaining train/ai={len(_list_files(train_ai))}'
    )
    purge_cache(old_paths + new_paths)

    info2 = detect(root)
    print('[verify]', info2)
    if info2['val_ai_identical_names'] or info2['wukong_val_ai_vqdm_named'] > 0:
        raise SystemExit('Fix failed: val/ai still looks like vqdm')
    print('[ok] wukong/val/ai rebuilt from wukong train holdout')
    print('[next] Re-pretrain party 8: set PRETRAIN_PARTY_IDS=[8] then python pretrain.py')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--root',
        default=None,
        help='GenImage root (default: Config.PRIVATE_DATASET_DIR)',
    )
    ap.add_argument('--val-size', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()

    if args.root:
        root = Path(args.root)
    else:
        from config import Config
        root = Path(Config.PRIVATE_DATASET_DIR)

    if args.check_only:
        info = detect(root)
        print(info)
        bad = info['val_ai_identical_names'] or info['wukong_val_ai_vqdm_named'] > 0
        sys.exit(1 if bad else 0)

    fix(root, args.val_size, args.seed, args.dry_run)


if __name__ == '__main__':
    main()
