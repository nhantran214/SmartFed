"""Local SSD image cache for GenImage / DiffusionFace.

First read pulls from the (slow) source drive, writes a resized JPEG into
``DATA_CACHE_DIR`` on NVMe; later reads hit the cache only.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Optional, Tuple

from PIL import Image, UnidentifiedImageError


def _cache_enabled() -> bool:
    try:
        from config import Config
        return bool(getattr(Config, 'DATA_CACHE_ENABLED', False))
    except Exception:
        return False


def _cache_dir() -> Optional[str]:
    try:
        from config import Config
        d = getattr(Config, 'DATA_CACHE_DIR', None)
        return os.path.expanduser(d) if d else None
    except Exception:
        return None


def _cache_size() -> Tuple[int, int]:
    try:
        from config import Config
        s = getattr(Config, 'DATA_CACHE_IMAGE_SIZE', 224)
        if isinstance(s, int):
            return (s, s)
        return tuple(s)
    except Exception:
        return (224, 224)


def cache_path_for(src_path: str, cache_dir: str) -> str:
    """Stable content-addressed path under cache_dir (avoids deep mirrors / collisions)."""
    abs_src = os.path.abspath(src_path)
    digest = hashlib.sha1(abs_src.encode('utf-8')).hexdigest()
    # Two-level fan-out keeps directories small
    return os.path.join(cache_dir, digest[:2], digest[2:4], digest + '.jpg')


def open_rgb_cached(src_path: str) -> Image.Image:
    """Open an RGB image, preferring a local SSD JPEG cache when enabled."""
    if not _cache_enabled():
        return _open_source_rgb(src_path)

    cache_dir = _cache_dir()
    if not cache_dir:
        return _open_source_rgb(src_path)

    cpath = cache_path_for(src_path, cache_dir)
    if os.path.isfile(cpath) and os.path.getsize(cpath) > 0:
        try:
            with Image.open(cpath) as img:
                return img.convert('RGB').copy()
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError):
            # Corrupt cache entry — rebuild below.
            try:
                os.remove(cpath)
            except OSError:
                pass

    img = _open_source_rgb(src_path)
    size = _cache_size()
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
    _atomic_save_jpeg(img, cpath)
    return img


def _open_source_rgb(src_path: str) -> Image.Image:
    if os.path.getsize(src_path) == 0:
        raise OSError(f'empty file: {src_path}')
    with Image.open(src_path) as img:
        return img.convert('RGB').copy()


def _atomic_save_jpeg(img: Image.Image, dest: str, quality: int = 90) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix='.jpg', dir=os.path.dirname(dest))
    os.close(fd)
    try:
        img.save(tmp, format='JPEG', quality=quality, optimize=True)
        os.replace(tmp, dest)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def prewarm_paths(paths, logger=None) -> Tuple[int, int]:
    """Populate cache for a list of source paths. Returns (ok, failed)."""
    ok = failed = 0
    for i, p in enumerate(paths):
        try:
            open_rgb_cached(p)
            ok += 1
        except Exception as e:
            failed += 1
            if logger is not None:
                logger.warning(f'cache prewarm failed: {p} ({e})')
        if logger is not None and (i + 1) % 2000 == 0:
            logger.info(f'Cache prewarm progress: {i + 1}/{len(paths)} (ok={ok}, fail={failed})')
    return ok, failed
