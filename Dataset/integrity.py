"""Dataset integrity helpers (GenImage / DiffusionFace folder layout)."""

from __future__ import annotations

import logging
import os
from typing import Optional


def check_genimage_client_val(datadir: str, logger: Optional[logging.Logger] = None) -> None:
    """Warn/raise if a client val/ai folder looks copied from another generator.

    Known issue: ``wukong/val/ai`` was a full duplicate of ``vqdm/val/ai``
    (filenames ``VQDM_*``), while ``wukong/train/ai`` correctly uses wukong fakes.
    Fix with ``scripts/fix_wukong_val.py``.
    """
    log = logger or logging.getLogger('smartfed')
    val_ai = os.path.join(datadir, 'val', 'ai')
    if not os.path.isdir(val_ai):
        return

    client = os.path.basename(os.path.abspath(datadir)).lower()
    try:
        names = [n for n in os.listdir(val_ai) if not n.startswith('.')]
    except OSError as e:
        log.warning(f'Dataset check: cannot list {val_ai}: {e}')
        return
    if not names:
        log.warning(f'Dataset check: empty val/ai under {datadir}')
        return

    sample = names[: min(50, len(names))]
    vqdm_like = sum(
        1 for n in sample
        if n.upper().startswith('VQDM') or '_vqdm_' in n.lower()
    )
    if client == 'wukong' and vqdm_like >= max(5, len(sample) // 2):
        msg = (
            f'Corrupt GenImage split: {datadir}/val/ai looks like VQDM fakes '
            f'({vqdm_like}/{len(sample)} sampled names). '
            f'Run: python scripts/fix_wukong_val.py && re-pretrain party 8.'
        )
        log.error(msg)
        raise RuntimeError(msg)

    # Soft check: val/ai prefix should usually mention the client name for most gens.
    if client in ('adm', 'biggan', 'glide', 'midjourney', 'sdv4', 'sdv5', 'wukong'):
        matched = sum(1 for n in sample if client in n.lower())
        if matched == 0 and vqdm_like > 0 and client != 'vqdm':
            log.warning(
                f'Dataset check: {client} val/ai has no "{client}" in filenames '
                f'but has VQDM-like names — possible wrong generator folder.'
            )
