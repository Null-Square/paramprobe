"""Routing regularizers for exponentially factorized parameter addresses."""

from __future__ import annotations

import math

import torch


def factorized_pair_collision(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Return the probability that two factorized Bernoulli addresses are equal.

    ``p`` and ``q`` have shape ``(..., r)`` and contain Bernoulli probabilities
    for ``r`` binary address factors. The result has shape ``(...)``. The
    computation is O(r) per pair and never materializes any of the ``2**r``
    composite addresses.
    """
    if p.shape != q.shape:
        raise ValueError("p and q must have identical shapes")
    if p.ndim < 1:
        raise ValueError("p and q must have at least one dimension")
    return (p * q + (1.0 - p) * (1.0 - q)).prod(dim=-1)


def exact_aggregate_collision(probabilities: torch.Tensor) -> torch.Tensor:
    """Exact collision probability of the batch-aggregated address distribution.

    For B samples and r factors this costs O(B^2 r), but is independent of the
    composite address count N = 2**r. It is intended for diagnostics and small
    reference experiments.
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (batch, factors)")
    p = probabilities[:, None, :]
    q = probabilities[None, :, :]
    return (p * q + (1.0 - p) * (1.0 - q)).prod(dim=-1).mean()


def paired_collision_estimate(probabilities: torch.Tensor) -> torch.Tensor:
    """O(B r) paired estimate of population full-address collision probability."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (batch, factors)")
    if probabilities.shape[0] < 2:
        raise ValueError("at least two samples are required")
    paired = torch.roll(probabilities, shifts=1, dims=0)
    return factorized_pair_collision(probabilities, paired).mean()


def renyi2_address_deficit(
    collision: torch.Tensor, num_factors: int
) -> torch.Tensor:
    """Return log(N*C2) = log(N) - H2 for N=2**num_factors.

    The value is zero for a uniform composite-address distribution and positive
    for collapsed distributions. This normalization makes the optimum invariant
    to the number of usable address factors.
    """
    if num_factors < 0:
        raise ValueError("num_factors must be non-negative")
    if num_factors == 0:
        return collision * 0.0
    return torch.log(collision.clamp_min(1e-12)) + num_factors * math.log(2.0)
