"""Streaming functional diagnostics for dense softmax local memories.

Argmax traffic is reported, but is never treated as a sufficient measure of
functional memory capacity. Statistics are training/evaluation instrumentation,
not free resident inference metadata. See the G10 mathematical note.
"""
from __future__ import annotations

import math
import numpy as np


def _probabilities(p: np.ndarray, slots: int) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != slots or len(p) == 0:
        raise ValueError('probabilities must have nonempty shape (queries, slots)')
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('probabilities must be finite and lie in [0, 1]')
    if not np.allclose(p.sum(axis=1), 1, rtol=1e-6, atol=1e-7):
        raise ValueError('probability rows must sum to one')
    return p


def participation_rank(gram: np.ndarray) -> float:
    """(trace G)^2 / trace(G^2), with zero mapped to zero (PSD G)."""
    g = np.asarray(gram, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or not np.isfinite(g).all():
        raise ValueError('finite square Gram matrix required')
    if not np.allclose(g, g.T, atol=1e-10):
        raise ValueError('Gram matrix must be symmetric')
    trace = float(np.trace(g))
    denominator = float(np.sum(g * g))
    if trace < -1e-10 or np.linalg.eigvalsh(g).min(initial=0) < -1e-10:
        raise ValueError('Gram matrix must be positive semidefinite')
    return 0.0 if denominator <= 0 else trace * trace / denominator


class StreamingSoftMemoryAudit:
    """Accumulate exact per-block probability mass and Gram statistics.

    Working memory is O(blocks * slots^2), independent of query count. Hard
    routing selects a global block; local attention inside that block is dense.
    """
    def __init__(self, blocks: int, slots: int) -> None:
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1
               for x in (blocks, slots)):
            raise ValueError('blocks and slots must be positive integers')
        self.blocks, self.slots = blocks, slots
        self.counts = np.zeros(blocks, dtype=np.int64)
        self.hard_counts = np.zeros((blocks, slots), dtype=np.int64)
        self.mass = np.zeros((blocks, slots), dtype=np.float64)
        self.gram = np.zeros((blocks, slots, slots), dtype=np.float64)
        self.entropy = np.zeros(blocks, dtype=np.float64)

    def update(self, block_ids: np.ndarray, probabilities: np.ndarray) -> None:
        p = _probabilities(probabilities, self.slots)
        ids = np.asarray(block_ids)
        if ids.ndim != 1 or len(ids) != len(p) or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('block_ids must be an integer vector aligned with queries')
        if np.any(ids < 0) or np.any(ids >= self.blocks):
            raise ValueError('block id outside configured store')
        for block in np.unique(ids):
            part = p[ids == block]
            self.counts[block] += len(part)
            self.hard_counts[block] += np.bincount(part.argmax(axis=1), minlength=self.slots)
            self.mass[block] += part.sum(axis=0)
            self.gram[block] += part.T @ part
            logp = np.zeros_like(part)
            np.log(part, out=logp, where=part > 0)
            self.entropy[block] += float(-(part * logp).sum())

    def summary(self) -> dict:
        total = int(self.counts.sum())
        if total == 0:
            raise ValueError('no observations')
        observed = self.counts > 0
        ranks, covariance_traces, soft_slots, per_block = [], [], [], []
        for block in np.flatnonzero(observed):
            n = int(self.counts[block])
            mean = self.mass[block] / n
            g = self.gram[block] / n
            covariance = g - np.outer(mean, mean)
            rank = participation_rank(g)
            variation = max(0.0, float(np.trace(covariance)))
            slots = 1.0 / float(mean @ mean)
            ranks.append(rank); covariance_traces.append(variation); soft_slots.append(slots)
            per_block.append({
                'block': int(block), 'queries': n,
                'hard_active_slots': int((self.hard_counts[block] > 0).sum()),
                'probability_mass': mean.tolist(),
                'value_gradient_gram_participation_rank': rank,
                'attention_covariance_trace': variation,
                'soft_mass_participation_slots': slots,
                'mean_attention_entropy_nats': float(self.entropy[block] / n),
            })
        weights = self.counts[observed] / total
        return {
            'queries': total,
            'observed_blocks': int(observed.sum()),
            'hard_dead_pair_fraction': float((self.hard_counts == 0).mean()),
            'weighted_gram_participation_rank': float(weights @ np.asarray(ranks)),
            'weighted_attention_covariance_trace': float(weights @ np.asarray(covariance_traces)),
            'weighted_soft_mass_participation_slots': float(weights @ np.asarray(soft_slots)),
            'mean_attention_entropy_nats': float(self.entropy.sum() / total),
            'per_block': per_block,
        }

    def constant_residual_mse_bound(self, block: int, values: np.ndarray, scale: float) -> float:
        """Empirical MSE upper bound for replacing s*tanh(PV) by s*tanh(mean(P)V).

        This is a residual-vector bound ON OBSERVED QUERIES. It is not an
        unseen-data bound or a downstream language-model cross-entropy bound.
        """
        if not isinstance(block, (int, np.integer)) or not 0 <= block < self.blocks:
            raise ValueError('invalid block')
        if self.counts[block] == 0:
            raise ValueError('block has no observations')
        v = np.asarray(values, dtype=np.float64)
        if v.ndim != 2 or v.shape[0] != self.slots or not np.isfinite(v).all():
            raise ValueError('values must have shape (slots, features) and be finite')
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('scale must be finite and nonnegative')
        n = self.counts[block]
        mean = self.mass[block] / n
        covariance = self.gram[block] / n - np.outer(mean, mean)
        return scale * scale * max(0.0, float(np.trace(v.T @ covariance @ v)))
