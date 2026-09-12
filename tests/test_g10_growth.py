import pytest
import torch
from torch import nn

from paramprobe.growth import clone_page_table, clone_adamw, two_route_risk


class TinyPages(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.num_pages = n
        self.w1 = nn.Embedding(n, 12, dtype=torch.float64)
        self.b1 = nn.Embedding(n, 3, dtype=torch.float64)
        self.w2 = nn.Embedding(n, 12, dtype=torch.float64)
        self.b2 = nn.Embedding(n, 4, dtype=torch.float64)

    def forward(self, ids, x):
        w1 = self.w1(ids).reshape(-1, 3, 4)
        h = torch.tanh(torch.bmm(w1, x[..., None]).squeeze(-1) + self.b1(ids))
        w2 = self.w2(ids).reshape(-1, 4, 3)
        return .15 * torch.tanh(torch.bmm(w2, h[..., None]).squeeze(-1) + self.b2(ids))


@pytest.mark.parametrize('factor', [1, 2, 16])
def test_clone_exact_for_all_child_routes_and_rng(factor):
    torch.manual_seed(21)
    old = TinyPages().eval()
    ids = torch.arange(3).repeat_interleave(factor)
    x = torch.randn(len(ids), 4, dtype=torch.float64)
    rng = torch.get_rng_state().clone()
    new = clone_page_table(old, factor)
    assert torch.equal(rng, torch.get_rng_state())
    assert new.num_pages == 3 * factor
    assert not new.training
    assert torch.equal(new(torch.arange(len(ids)), x), old(ids, x))
    for a, b in zip(old.parameters(), new.parameters()):
        assert a.data_ptr() != b.data_ptr()
    with torch.no_grad():
        new.b2.weight[0].add_(1)
    assert not torch.equal(new.b2.weight[0], old.b2.weight[0])
    if factor > 1:
        assert torch.equal(new.b2.weight[1], old.b2.weight[0])


@pytest.mark.parametrize('factor', [0, -1, 1.5, True])
def test_invalid_factor(factor):
    with pytest.raises(ValueError):
        clone_page_table(TinyPages(), factor)


def test_optimizer_moments_copied_without_aliasing():
    torch.manual_seed(22)
    old = TinyPages()
    opt = torch.optim.AdamW(old.parameters(), lr=.007, amsgrad=True)
    loss = old(torch.tensor([0, 1]), torch.randn(2, 4, dtype=torch.float64)).square().sum()
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    new = clone_page_table(old, 2)
    new_opt = clone_adamw(opt, old, new, 2)
    assert new_opt.param_groups[0]['lr'] == .007
    for old_p, new_p in zip(old.parameters(), new.parameters()):
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            expected = opt.state[old_p][name].repeat_interleave(2, dim=0)
            assert torch.equal(new_opt.state[new_p][name], expected)
            assert new_opt.state[new_p][name].data_ptr() != opt.state[old_p][name].data_ptr()
        assert torch.equal(new_opt.state[new_p]['step'], opt.state[old_p]['step'])
    new_opt.zero_grad()
    new(torch.tensor([0, 3]), torch.randn(2, 4, dtype=torch.float64)).sum().backward()
    new_opt.step()


def test_route_gradient_is_expected_risk_not_winner_classification():
    z = torch.tensor([-.7, .2], dtype=torch.float64, requires_grad=True)
    left = torch.tensor([1., 3.], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([2., .5], dtype=torch.float64, requires_grad=True)
    two_route_risk(z, left, right).backward()
    p = torch.sigmoid(z.detach())
    assert torch.allclose(z.grad, p * (1-p) * (right.detach()-left.detach()) / 2)
    assert left.grad is None and right.grad is None


def test_identical_children_have_zero_router_task_gradient():
    z = torch.linspace(-3, 3, 9, requires_grad=True)
    loss = torch.linspace(.2, 3, 9)
    two_route_risk(z, loss, loss).backward()
    assert torch.equal(z.grad, torch.zeros_like(z))


def test_changed_gate_breaks_function_preservation():
    torch.manual_seed(23)
    old = TinyPages()
    new = clone_page_table(old, 2)
    x = torch.randn(3, 4, dtype=torch.float64)
    y = old(torch.arange(3), x)
    y2 = new(torch.arange(3) * 2, x)
    assert torch.equal(y, y2)
    assert not torch.allclose(y, .5 * y2)


@pytest.mark.parametrize('bad', [torch.tensor([]), torch.tensor([float('nan')])])
def test_invalid_risk_input(bad):
    with pytest.raises(ValueError):
        two_route_risk(bad, bad, bad)
