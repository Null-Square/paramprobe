import itertools

import torch

from paramprobe.routing_regularizers import (
    exact_aggregate_collision,
    factorized_pair_collision,
    renyi2_address_deficit,
)


def _explicit_collision(probabilities: torch.Tensor) -> torch.Tensor:
    factors = probabilities.shape[1]
    masses = []
    for address in itertools.product([0, 1], repeat=factors):
        bits = torch.tensor(address, dtype=probabilities.dtype)
        per_sample = torch.where(
            bits.bool(), probabilities, 1.0 - probabilities
        ).prod(dim=1)
        masses.append(per_sample.mean())
    aggregate = torch.stack(masses)
    return torch.sum(aggregate**2)


def test_factorized_collision_matches_explicit_composite_distribution():
    torch.manual_seed(7)
    probabilities = 0.1 + 0.8 * torch.rand(9, 5)
    efficient = exact_aggregate_collision(probabilities)
    explicit = _explicit_collision(probabilities)
    torch.testing.assert_close(efficient, explicit, atol=1e-7, rtol=1e-7)


def test_uniform_distribution_has_zero_renyi2_deficit():
    probabilities = torch.full((11, 4), 0.5)
    collision = exact_aggregate_collision(probabilities)
    torch.testing.assert_close(collision, torch.tensor(1.0 / 16.0))
    torch.testing.assert_close(
        renyi2_address_deficit(collision, 4), torch.tensor(0.0), atol=1e-7, rtol=0
    )


def test_pair_collision_identity():
    p = torch.tensor([[0.2, 0.7]])
    q = torch.tensor([[0.3, 0.4]])
    expected = (0.2 * 0.3 + 0.8 * 0.7) * (0.7 * 0.4 + 0.3 * 0.6)
    torch.testing.assert_close(
        factorized_pair_collision(p, q), torch.tensor([expected])
    )
