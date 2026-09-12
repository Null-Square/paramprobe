"""Function-preserving page expansion and task-risk routing utilities.

These are supporting mechanisms, not claims of new general learning theory.
See docs/g10_mathematical_guarantees.md for the assumptions and proofs.
"""
from __future__ import annotations

import copy
import inspect
from typing import TypeVar

import torch
from torch import nn

PageTable = TypeVar('PageTable', bound=nn.Module)
_FIELDS = ('w1', 'b1', 'w2', 'b2')


def clone_page_table(pages: PageTable, factor: int) -> PageTable:
    """Replace every page by ``factor`` independent, initially identical copies.

    New route ids must satisfy new_id // factor == old_id. The residual gate
    and downstream computation must remain unchanged. This function consumes
    no RNG state. It is training-side work; inference still selects one page.
    """
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError('factor must be a positive integer')
    n = getattr(pages, 'num_pages', None)
    if not isinstance(n, int) or n < 1:
        raise TypeError('pages must expose a positive integer num_pages')
    for name in _FIELDS:
        layer = getattr(pages, name, None)
        if not isinstance(layer, nn.Embedding) or layer.num_embeddings != n:
            raise TypeError(f'{name} must be an nn.Embedding with num_pages rows')
        if layer.padding_idx is not None or layer.max_norm is not None:
            raise ValueError('padding and max_norm embeddings are not supported')
    if set(dict(pages.named_parameters())) != {f'{n}.weight' for n in _FIELDS}:
        raise ValueError('only the four supported page parameter tables are allowed')
    grown = copy.deepcopy(pages)
    for name in _FIELDS:
        old = getattr(pages, name)
        weight = old.weight.detach().repeat_interleave(factor, dim=0).clone()
        new = nn.Embedding.from_pretrained(
            weight, freeze=not old.weight.requires_grad,
            scale_grad_by_freq=old.scale_grad_by_freq, sparse=old.sparse,
        )
        new.train(old.training)
        setattr(grown, name, new)
    grown.num_pages = n * factor
    return grown


def clone_adamw(
    optimizer: torch.optim.AdamW,
    source: nn.Module,
    target: nn.Module,
    factor: int,
) -> torch.optim.AdamW:
    """Copy an AdamW optimizer and its moments to a cloned page table.

    Copying moments avoids silently resetting the warm-start optimizer. It
    does NOT imply identical future updates once children receive different
    data, nor does it prove that this optimizer policy is optimal.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError('only AdamW state copying is supported')
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError('factor must be a positive integer')
    old_named, new_named = dict(source.named_parameters()), dict(target.named_parameters())
    if old_named.keys() != new_named.keys():
        raise ValueError('source and target parameter names differ')
    by_id = {id(p): name for name, p in old_named.items()}
    groups = []
    seen = []
    for group in optimizer.param_groups:
        names = [by_id.get(id(p)) for p in group['params']]
        if any(name is None for name in names):
            raise ValueError('optimizer includes parameters outside source')
        seen.extend(names)
        item = {k: copy.deepcopy(v) for k, v in group.items() if k != 'params'}
        item['params'] = [new_named[name] for name in names]
        groups.append(item)
    if len(seen) != len(old_named) or set(seen) != set(old_named):
        raise ValueError('optimizer must cover every source parameter exactly once')
    for name, old in old_named.items():
        new = new_named[name]
        if old.ndim != 2 or new.shape != (old.shape[0] * factor, old.shape[1]):
            raise ValueError(f'non-cloned shape for {name}')
    accepted = inspect.signature(torch.optim.AdamW).parameters
    defaults = {k: copy.deepcopy(v) for k, v in optimizer.defaults.items() if k in accepted}
    result = torch.optim.AdamW(groups, **defaults)
    for name, old in old_named.items():
        if old not in optimizer.state:
            continue
        state = {}
        for key, value in optimizer.state[old].items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    state[key] = value.detach().clone()
                elif value.shape == old.shape:
                    state[key] = value.detach().repeat_interleave(factor, dim=0).clone()
                else:
                    raise ValueError(f'unsupported optimizer state {key}: {value.shape}')
            else:
                state[key] = copy.deepcopy(value)
        result.state[new_named[name]] = state
    return result


def two_route_risk(
    prefix_logit: torch.Tensor,
    left_loss: torch.Tensor,
    right_loss: torch.Tensor,
) -> torch.Tensor:
    """Mean expected risk of a prefix-only binary stochastic route.

    The two losses may use TRAINING labels. They are detached so this function
    updates only the router. At inference the router receives only the prefix.
    Both downstream candidate evaluations are training cost and must be counted.
    """
    if prefix_logit.shape != left_loss.shape or left_loss.shape != right_loss.shape:
        raise ValueError('logits and candidate losses must have identical shapes')
    if prefix_logit.numel() == 0:
        raise ValueError('empty routing batch')
    for x in (prefix_logit, left_loss, right_loss):
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('finite floating point tensors required')
    p = torch.sigmoid(prefix_logit)
    return ((1 - p) * left_loss.detach() + p * right_loss.detach()).mean()
