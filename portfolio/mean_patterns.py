from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import List, Tuple
import numpy as np

PatternResult = Tuple[np.ndarray, List[List[int]]]


def _source_mu_high(k: int, mu_high: float, mu_high_growth: float) -> float:
    """Per-source 'high' mean. k is the 0-indexed source position, so source
    0 gets exactly mu_high and later sources get progressively more extreme
    means when mu_high_growth != 0. Using an
    absolute (not K-normalized) offset keeps the schedule prefix-consistent:
    source k's mean is the same regardless of how large the total K sweep is.
    """
    return float(mu_high) + k * float(mu_high_growth)


def _source_mu_low(k: int, mu_low: float, mu_low_shrink: float) -> float:
    """Per-source 'low' mean, mirroring _source_mu_high. mu_low_shrink >= 0
    pushes the low value further negative (away from zero) for later sources."""
    return float(mu_low) - k * float(mu_low_shrink)


def make_combinatorial_overlap_blocks_pattern(
    K: int,
    d: int,
    block_size: int = 4,
    stride: int = 2,
    mu_high: float = 0.004,
    mu_low: float = -0.004,
    mu_high_growth: float = 0.0,
    mu_low_shrink: float = 0.0,
) -> PatternResult:
    overlap_size = max(0, block_size - stride)
    all_blocks = [tuple(block) for block in combinations(range(d), block_size)]
    if K > len(all_blocks):
        raise ValueError(
            f'combinatorial_overlap_blocks supports at most {len(all_blocks)} distinct blocks '
            f'for d={d} and block_size={block_size}, but received K={K}'
        )

    means = np.zeros((K, d), dtype=float)
    if K == 0:
        return means, []

    groups: List[List[int]] = [list(all_blocks[0])]
    used = {all_blocks[0]}
    asset_counts = Counter(all_blocks[0])

    for _ in range(1, K):
        prev = set(groups[-1])
        candidates: List[tuple[tuple[float, float, tuple[int, ...]], tuple[int, ...]]] = []
        for block in all_blocks:
            if block in used:
                continue
            block_set = set(block)
            if len(prev & block_set) != overlap_size:
                continue
            # Among the unused blocks that share overlap_size assets with the previous
            # block, prefer the one whose assets were used most often so far, then the
            # one with the largest total overlap with earlier blocks, then the
            # lexicographically smallest one.
            score = (
                -float(sum(asset_counts[idx] for idx in block)),
                -float(sum(len(block_set & set(old_group)) for old_group in groups[:-1])),
                block,
            )
            candidates.append((score, block))

        if not candidates:
            raise ValueError(
                'Could not construct enough distinct combinatorial overlap blocks '
                f'with d={d}, block_size={block_size}, and overlap_size={overlap_size}.'
            )

        _, chosen_block = min(candidates, key=lambda item: item[0])
        groups.append(list(chosen_block))
        used.add(chosen_block)
        asset_counts.update(chosen_block)

    for k, group in enumerate(groups):
        means[k, :] = _source_mu_low(k, mu_low, mu_low_shrink)
        means[k, group] = _source_mu_high(k, mu_high, mu_high_growth)
    return means, groups
