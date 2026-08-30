"""Factorized addressing utilities.

The first implementation deliberately uses a best-first exact top-k search over
an additive Cartesian score. It never enumerates the full m**r address space.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Sequence

import numpy as np


def mixed_radix_id(indices: Sequence[int], radix: int) -> int:
    """Map an r-factor address to a dense integer id in [0, radix**r)."""
    if radix < 1:
        raise ValueError("radix must be positive")
    out = 0
    for idx in indices:
        if idx < 0 or idx >= radix:
            raise ValueError(f"index {idx} outside radix {radix}")
        out = out * radix + int(idx)
    return out


@dataclass(frozen=True)
class RoutedAddress:
    score: float
    factors: tuple[int, ...]

    def block_id(self, radix: int) -> int:
        return mixed_radix_id(self.factors, radix)


class FactorizedTopKRouter:
    """Exact top-k router for additive factor scores.

    Each factor owns ``m`` keys in R^s. A query supplies one vector per factor.
    Address score is the sum of per-factor dot products. Exact top-k composite
    addresses are obtained with best-first search over the sorted score lattice.
    """

    def __init__(self, keys: np.ndarray):
        keys = np.asarray(keys, dtype=np.float64)
        if keys.ndim != 3:
            raise ValueError("keys must have shape [r, m, s]")
        self.keys = keys
        self.r, self.m, self.s = keys.shape

    @property
    def address_space(self) -> int:
        return self.m ** self.r

    @property
    def metadata_scalars(self) -> int:
        return int(self.keys.size)

    def factor_scores(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float64)
        if query.shape != (self.r, self.s):
            raise ValueError(f"query must have shape {(self.r, self.s)}")
        return np.einsum("rms,rs->rm", self.keys, query)

    def topk(self, query: np.ndarray, k: int) -> list[RoutedAddress]:
        if k < 1:
            raise ValueError("k must be positive")
        if k > self.address_space:
            raise ValueError("k exceeds address space")

        scores = self.factor_scores(query)
        order = np.argsort(-scores, axis=1, kind="stable")
        sorted_scores = np.take_along_axis(scores, order, axis=1)

        start = (0,) * self.r
        heap: list[tuple[float, tuple[int, ...]]] = [
            (-float(sorted_scores[:, 0].sum()), start)
        ]
        seen = {start}
        out: list[RoutedAddress] = []

        while heap and len(out) < k:
            neg_score, ranks = heapq.heappop(heap)
            factors = tuple(int(order[j, ranks[j]]) for j in range(self.r))
            out.append(RoutedAddress(score=-neg_score, factors=factors))

            for j in range(self.r):
                if ranks[j] + 1 >= self.m:
                    continue
                nxt = list(ranks)
                nxt[j] += 1
                nxt_t = tuple(nxt)
                if nxt_t in seen:
                    continue
                seen.add(nxt_t)
                nxt_score = sum(sorted_scores[d, nxt_t[d]] for d in range(self.r))
                heapq.heappush(heap, (-float(nxt_score), nxt_t))

        return out


def brute_force_topk(factor_scores: np.ndarray, k: int) -> list[RoutedAddress]:
    """Reference implementation for tiny tests only."""
    import itertools

    factor_scores = np.asarray(factor_scores, dtype=np.float64)
    r, m = factor_scores.shape
    values: list[RoutedAddress] = []
    for factors in itertools.product(range(m), repeat=r):
        score = sum(float(factor_scores[j, factors[j]]) for j in range(r))
        values.append(RoutedAddress(score=score, factors=tuple(factors)))
    values.sort(key=lambda item: (-item.score, item.factors))
    return values[:k]
