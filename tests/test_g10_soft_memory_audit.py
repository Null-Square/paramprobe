import numpy as np
import pytest
import torch

from paramprobe.soft_memory_audit import StreamingSoftMemoryAudit, participation_rank


def test_never_argmax_can_have_full_rank_and_nonzero_gradient():
    p = np.array([[.6, .3, .1], [.6, .1, .3], [.7, .2, .1]])
    audit = StreamingSoftMemoryAudit(1, 3)
    audit.update(np.zeros(3, dtype=int), p)
    assert np.linalg.matrix_rank(p) == 3
    assert audit.summary()['hard_dead_pair_fraction'] == pytest.approx(2/3)
    values = torch.ones(3, 2, dtype=torch.float64, requires_grad=True)
    (torch.tensor(p) @ values).sum().backward()
    assert torch.all(values.grad > 0)


def test_perfect_hard_usage_can_have_arbitrarily_small_functional_variation():
    m, eps = 21, 1e-7
    p = (1-eps) * np.ones((m, m)) / m + eps * np.eye(m)
    audit = StreamingSoftMemoryAudit(1, m)
    audit.update(np.zeros(m, dtype=int), p)
    s = audit.summary()
    assert s['hard_dead_pair_fraction'] == 0
    assert s['weighted_gram_participation_rank'] == pytest.approx(1., abs=1e-10)
    assert s['weighted_attention_covariance_trace'] < 1e-12


def test_uniform_attention_has_balanced_mass_but_rank_one():
    p = np.full((40, 21), 1/21)
    audit = StreamingSoftMemoryAudit(1, 21)
    audit.update(np.zeros(40, dtype=int), p)
    s = audit.summary()
    assert s['weighted_soft_mass_participation_slots'] == pytest.approx(21)
    assert s['weighted_gram_participation_rank'] == pytest.approx(1)
    assert s['weighted_attention_covariance_trace'] < 1e-14


def test_streaming_equals_single_update_and_bound_holds():
    rng = np.random.default_rng(26)
    p = rng.dirichlet(np.ones(5), size=100)
    ids = rng.integers(0, 3, size=100)
    one, split = StreamingSoftMemoryAudit(3, 5), StreamingSoftMemoryAudit(3, 5)
    one.update(ids, p)
    for indices in np.array_split(np.arange(100), 7):
        split.update(ids[indices], p[indices])
    assert np.array_equal(one.counts, split.counts)
    assert np.array_equal(one.hard_counts, split.hard_counts)
    assert np.allclose(one.mass, split.mass)
    assert np.allclose(one.gram, split.gram)
    v = rng.normal(size=(5, 9))
    for block in range(3):
        probs = p[ids == block]
        out = .15 * np.tanh(probs @ v)
        constant = .15 * np.tanh(probs.mean(axis=0) @ v)
        mse = np.mean(np.sum((out-constant)**2, axis=1))
        assert mse <= one.constant_residual_mse_bound(block, v, .15) + 1e-12


def test_zero_or_equal_values_block_key_gradients():
    torch.manual_seed(27)
    for value in (0., 2.):
        k = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
        v = torch.full((5, 7), value, dtype=torch.float64, requires_grad=True)
        x = torch.randn(11, 4, dtype=torch.float64)
        out = .15 * torch.tanh(torch.softmax(x @ k.T, dim=-1) @ v)
        out.sum().backward()
        assert torch.allclose(k.grad, torch.zeros_like(k), atol=1e-14)
        assert torch.any(v.grad != 0)


def test_probability_gram_is_value_jacobian_gram():
    torch.manual_seed(28)
    p = torch.softmax(torch.randn(7, 3, dtype=torch.float64), -1)
    v = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    j = torch.autograd.functional.jacobian(lambda a: (p @ a).reshape(-1), v).reshape(14, 6)
    assert torch.allclose(j.T @ j, torch.kron(p.T @ p, torch.eye(2, dtype=torch.float64)))


@pytest.mark.parametrize('p', [np.array([[.2,.2]]), np.array([[1.1,-.1]]), np.array([[np.nan,1.]])])
def test_invalid_probabilities_fail(p):
    audit = StreamingSoftMemoryAudit(1, 2)
    with pytest.raises(ValueError):
        audit.update(np.array([0]), p)


def test_invalid_indices_and_empty_summary():
    a = StreamingSoftMemoryAudit(1, 2)
    with pytest.raises(ValueError): a.summary()
    with pytest.raises(ValueError): a.update(np.array([1]), np.array([[.5,.5]]))
    with pytest.raises(ValueError): a.update(np.array([0.]), np.array([[.5,.5]]))
    with pytest.raises(ValueError): participation_rank(np.array([[1., 2.], [2., 1.]]))
