"""Helpers for parallel client train/eval on a single GPU.

PyTorch releases the GIL during CUDA kernels, so a small ThreadPoolExecutor
can overlap multiple client workloads when they fit in VRAM together.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def iter_chunks(items: Sequence[T], chunk_size: int) -> Iterable[Sequence[T]]:
    """Yield contiguous chunks of ``items`` with size ``chunk_size``."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def run_parallel(
    fns: Sequence[Callable[[], R]],
    max_workers: int,
) -> List[R]:
    """Run callables concurrently and return results in the original order."""
    if not fns:
        return []
    workers = max(1, min(max_workers, len(fns)))
    if workers == 1:
        return [fn() for fn in fns]

    results: List[R] = [None] * len(fns)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {executor.submit(fn): idx for idx, fn in enumerate(fns)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results


def map_parallel(
    fn: Callable[[T], R],
    items: Sequence[T],
    max_workers: int,
) -> List[R]:
    """Map ``fn`` over ``items`` with up to ``max_workers`` threads."""
    return run_parallel([lambda item=item: fn(item) for item in items], max_workers)
