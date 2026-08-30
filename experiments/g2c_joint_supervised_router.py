"""G2c: jointly train a fixed router and page-local nonlinear operators.

This is a controlled bridge between separate routing/operator experiments and a
fully task-driven sparse architecture. Router and page MLPs are optimized in the
same run, hard one-page routing is used for the task forward pass, and resource
budgets are fixed across the capacity sweep.

Important limitation: the discrete page lookup blocks task gradients into the
router, so the router is trained with known semantic address bits while the page
operators are trained from task loss. This gate tests co-training stability,
page utilization, and routing noise; it does NOT yet demonstrate task-only route
learning. That is the next gate.
"""

from __future__ import annotations

import argparse
import math
import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ints_to_bits(values: np.ndarray, width: int) -> np.ndarray:
    shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
    return ((values[:, None] >> shifts) & 1).astype(np.float32)


def bits_to_page(bits: torch.Tensor, used_bits: int) -> torch.Tensor:
    if used_bits == 0:
        return torch.zeros(bits.shape[0], dtype=torch.long, device=bits.device)
    ids = torch.zeros(bits.shape[0], dtype=torch.long, device=bits.device)
    for j in range(used_bits):
        ids = (ids << 1) | bits[:, j].long()
    return ids


class FixedRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, max_bits: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_bits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PageMLPTable(nn.Module):
    def __init__(
        self,
        num_pages: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.w1 = nn.Embedding(num_pages, hidden_dim * input_dim)
        self.b1 = nn.Embedding(num_pages, hidden_dim)
        self.w2 = nn.Embedding(num_pages, output_dim * hidden_dim)
        self.b2 = nn.Embedding(num_pages, output_dim)
        nn.init.normal_(self.w1.weight, std=0.25 / math.sqrt(input_dim))
        nn.init.normal_(self.w2.weight, std=0.25 / math.sqrt(hidden_dim))
        nn.init.zeros_(self.b1.weight)
        nn.init.zeros_(self.b2.weight)

    @property
    def page_parameters(self) -> int:
        return (
            self.hidden_dim * self.input_dim
            + self.hidden_dim
            + self.output_dim * self.hidden_dim
            + self.output_dim
        )

    def forward(self, page_ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        w1 = self.w1(page_ids).view(batch, self.hidden_dim, self.input_dim)
        h = torch.tanh(
            torch.bmm(w1, x.unsqueeze(-1)).squeeze(-1) + self.b1(page_ids)
        )
        w2 = self.w2(page_ids).view(batch, self.output_dim, self.hidden_dim)
        return torch.bmm(w2, h.unsqueeze(-1)).squeeze(-1) + self.b2(page_ids)


def make_teacher(
    num_items: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w1 = torch.from_numpy(
        rng.normal(
            scale=0.7 / math.sqrt(input_dim),
            size=(num_items, hidden_dim, input_dim),
        ).astype(np.float32)
    )
    b1 = torch.from_numpy(
        rng.normal(scale=0.2, size=(num_items, hidden_dim)).astype(np.float32)
    )
    w2 = torch.from_numpy(
        rng.normal(
            scale=0.8 / math.sqrt(hidden_dim),
            size=(num_items, output_dim, hidden_dim),
        ).astype(np.float32)
    )
    b2 = torch.from_numpy(
        rng.normal(scale=0.1, size=(num_items, output_dim)).astype(np.float32)
    )
    return w1, b1, w2, b2


def teacher_forward(
    params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    item_ids: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    w1, b1, w2, b2 = params
    h = torch.tanh(
        torch.bmm(w1[item_ids], x.unsqueeze(-1)).squeeze(-1) + b1[item_ids]
    )
    return torch.bmm(w2[item_ids], h.unsqueeze(-1)).squeeze(-1) + b2[item_ids]


def normalized_entropy(counts: np.ndarray) -> float:
    positive = counts[counts > 0].astype(np.float64)
    if len(counts) <= 1:
        return 1.0
    probs = positive / positive.sum()
    return float(-(probs * np.log(probs)).sum() / np.log(len(counts)))


def run_one(args: argparse.Namespace, used_bits: int) -> dict[str, float]:
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.num_items != 2**args.max_bits:
        raise ValueError("G2c currently uses the full binary semantic address space")

    addresses = np.arange(args.num_items, dtype=np.int64)
    semantic_bits = ints_to_bits(addresses, args.max_bits)
    signs = semantic_bits * 2.0 - 1.0
    encoder = rng.normal(size=(args.max_bits, args.context_dim)).astype(np.float32)
    encoder /= np.linalg.norm(encoder, axis=1, keepdims=True) + 1e-12
    base_context = (signs @ encoder) / math.sqrt(args.max_bits)
    base_context += rng.normal(
        scale=args.item_nuisance, size=base_context.shape
    ).astype(np.float32)

    teacher = make_teacher(
        args.num_items,
        args.query_dim,
        args.page_hidden_dim,
        args.output_dim,
        rng,
    )

    train_items_np = np.repeat(np.arange(args.num_items), args.train_queries_per_item)
    val_items_np = np.repeat(np.arange(args.num_items), args.val_queries_per_item)
    train_items = torch.from_numpy(train_items_np).long()
    val_items = torch.from_numpy(val_items_np).long()

    train_context = np.repeat(base_context, args.train_queries_per_item, axis=0)
    train_context += rng.normal(
        scale=args.context_noise, size=train_context.shape
    ).astype(np.float32)
    val_context = np.repeat(base_context, args.val_queries_per_item, axis=0)
    val_context += rng.normal(
        scale=args.context_noise, size=val_context.shape
    ).astype(np.float32)

    x_train = torch.from_numpy(
        rng.normal(size=(len(train_items_np), args.query_dim)).astype(np.float32)
    )
    x_val = torch.from_numpy(
        rng.normal(size=(len(val_items_np), args.query_dim)).astype(np.float32)
    )
    with torch.no_grad():
        y_train = teacher_forward(teacher, train_items, x_train)
        y_val = teacher_forward(teacher, val_items, x_val)

    context_train_t = torch.from_numpy(train_context)
    context_val_t = torch.from_numpy(val_context)
    bits_train_t = torch.from_numpy(semantic_bits[train_items_np])
    bits_val_t = torch.from_numpy(semantic_bits[val_items_np])

    num_pages = 1 if used_bits == 0 else 1 << used_bits
    router = FixedRouter(args.context_dim, args.router_hidden_dim, args.max_bits)
    pages = PageMLPTable(
        num_pages,
        args.query_dim,
        args.page_hidden_dim,
        args.output_dim,
    )
    page_payload_bytes = pages.page_parameters * 4
    if page_payload_bytes > args.block_bytes:
        raise ValueError("page MLP does not fit within block_bytes")

    optimizer = torch.optim.Adam(
        [
            {"params": router.parameters(), "lr": args.router_lr},
            {"params": pages.parameters(), "lr": args.page_lr},
        ]
    )
    update_counts = torch.zeros(num_pages, dtype=torch.long)

    for _ in range(args.epochs):
        order = torch.randperm(len(train_items))
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            logits = router(context_train_t[idx])
            hard_bits = logits > 0
            selected_pages = bits_to_page(hard_bits, used_bits)
            prediction = pages(selected_pages, x_train[idx])
            task_loss = torch.mean((prediction - y_train[idx]) ** 2)
            route_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, bits_train_t[idx]
            )
            loss = task_loss + args.route_loss_weight * route_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_counts += torch.bincount(
                selected_pages.detach().cpu(), minlength=num_pages
            )

    router.eval()
    pages.eval()
    with torch.no_grad():
        logits = router(context_val_t)
        hard_bits = logits > 0
        selected_pages = bits_to_page(hard_bits, used_bits)
        prediction = pages(selected_pages, x_val)
        target_energy = torch.mean(y_val**2)
        normalized_mse = float(torch.mean((prediction - y_val) ** 2) / target_energy)
        bit_accuracy = float((hard_bits == bits_val_t.bool()).float().mean())
        full_address_accuracy = float(
            (hard_bits == bits_val_t.bool()).all(dim=1).float().mean()
        )
        prefix_accuracy = (
            1.0
            if used_bits == 0
            else float(
                (hard_bits[:, :used_bits] == bits_val_t[:, :used_bits].bool())
                .all(dim=1)
                .float()
                .mean()
            )
        )
        val_counts = torch.bincount(
            selected_pages.cpu(), minlength=num_pages
        ).numpy()

    return {
        "used_bits": float(used_bits),
        "pages": float(num_pages),
        "normalized_mse": normalized_mse,
        "prefix_accuracy": prefix_accuracy,
        "bit_accuracy": bit_accuracy,
        "full_address_accuracy": full_address_accuracy,
        "utilization_entropy": normalized_entropy(val_counts),
        "dead_page_fraction": float(np.mean(val_counts == 0)),
        "never_updated_page_fraction": float(np.mean(update_counts.numpy() == 0)),
        "page_payload_bytes": float(page_payload_bytes),
    }


def run(args: argparse.Namespace) -> None:
    controller_params = (
        args.context_dim * args.router_hidden_dim
        + args.router_hidden_dim
        + args.router_hidden_dim * args.max_bits
        + args.max_bits
    )
    routing_macs = (
        args.context_dim * args.router_hidden_dim
        + args.router_hidden_dim * args.max_bits
    )
    operator_macs = (
        args.query_dim * args.page_hidden_dim
        + args.page_hidden_dim * args.output_dim
    )

    print("# ParamProbe G2c: joint supervised routing + page MLP training")
    print(f"seed={args.seed}")
    print(f"controller_params={controller_params}")
    print(f"routing_macs_per_query={routing_macs}")
    print(f"active_operator_macs_per_query={operator_macs}")
    print("q=1")
    print(f"block_bytes={args.block_bytes}")
    print(f"logical_bytes_per_query={args.block_bytes}")
    print("router_task_gradient=none_discrete_lookup")
    print("router_supervision=semantic_address_bits")
    print()
    print(
        "used_bits\tpages\texternal_MiB\tnmse\tprefix_acc\tbit_acc\t"
        "full_acc\tutil_entropy\tdead_pages\tnever_updated"
    )

    sweep = [0] + list(range(args.min_bits, args.max_bits + 1, args.bit_step))
    if sweep[-1] != args.max_bits:
        sweep.append(args.max_bits)
    for used_bits in sweep:
        result = run_one(args, used_bits)
        external_mib = result["pages"] * args.block_bytes / (1024**2)
        print(
            f"{used_bits}\t{int(result['pages'])}\t{external_mib:.6f}\t"
            f"{result['normalized_mse']:.8f}\t{result['prefix_accuracy']:.8f}\t"
            f"{result['bit_accuracy']:.8f}\t{result['full_address_accuracy']:.8f}\t"
            f"{result['utilization_entropy']:.8f}\t{result['dead_page_fraction']:.8f}\t"
            f"{result['never_updated_page_fraction']:.8f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-items", type=int, default=256)
    parser.add_argument("--max-bits", type=int, default=8)
    parser.add_argument("--min-bits", type=int, default=2)
    parser.add_argument("--bit-step", type=int, default=2)
    parser.add_argument("--context-dim", type=int, default=48)
    parser.add_argument("--router-hidden-dim", type=int, default=96)
    parser.add_argument("--query-dim", type=int, default=8)
    parser.add_argument("--page-hidden-dim", type=int, default=32)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--train-queries-per-item", type=int, default=64)
    parser.add_argument("--val-queries-per-item", type=int, default=24)
    parser.add_argument("--context-noise", type=float, default=0.12)
    parser.add_argument("--item-nuisance", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--router-lr", type=float, default=2e-3)
    parser.add_argument("--page-lr", type=float, default=1e-2)
    parser.add_argument("--route-loss-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    run(parser.parse_args())
