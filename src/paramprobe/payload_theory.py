"""Algebraic controls, not novel architecture claims or global LM guarantees."""
from __future__ import annotations
import torch


def tanh_via_paired_softmax(z: torch.Tensor) -> torch.Tensor:
    """Each hidden coordinate has its OWN two-slot normalization."""
    p = torch.softmax(torch.stack((z, -z), dim=-1),dim=-1)
    return p[...,0] - p[...,1]


def operator(x, w1, b1, w2, b2, paired: bool = False, scale: float = 0.15):
    z = x @ w1.T + b1
    a = tanh_via_paired_softmax(z) if paired else torch.tanh(z)
    return scale * torch.tanh(a @ w2.T + b2)


def pruning_bound(probabilities, values, retained, scale=0.15):
    """Per-query Euclidean error bound, requiring ORIGINAL probabilities.

    Not a cheap deployed confidence certificate: dropped keys are needed to
    compute those probabilities. This is a mathematical audit bound.
    """
    if probabilities.ndim != 2 or values.ndim != 2 or probabilities.shape[1] != values.shape[0]:
        raise ValueError("shape mismatch")
    if not torch.isfinite(probabilities).all() or not torch.isfinite(values).all():
        raise ValueError("nonfinite input")
    if (probabilities < 0).any() or not torch.allclose(probabilities.sum(-1),torch.ones_like(probabilities[:,0])):
        raise ValueError("probabilities must be normalized")
    if retained.ndim != 1 or retained.numel() == 0 or retained.dtype != torch.long:
        raise ValueError("retained must be nonempty long vector")
    if int(retained.min()) < 0 or int(retained.max()) >= values.shape[0] or retained.unique().numel() != retained.numel():
        raise ValueError("invalid retained indices")
    mass = probabilities[:,retained].sum(-1)
    if (mass <= 0).any(): raise ValueError("retained mass is zero")
    eps = (1-mass).clamp(min=0)
    return 2 * abs(scale) * eps * torch.linalg.vector_norm(values,dim=-1).max()
