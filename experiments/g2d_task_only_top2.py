"""G2d: task-only discrete routing with top-2 training / top-1 inference.

No semantic page-address labels are provided. The fixed-size binary router must
learn page specialization from task loss alone.

For independent Bernoulli address factors, the highest-scoring page is the hard
sign pattern of the router logits. The exact second-best page differs by flipping
the least-confident active bit. During training we evaluate those two pages and
softly mix their outputs, which gives task gradients to the selected router
scores. At inference only the best page is read: q_infer=1.

This is intentionally a falsification precheck. With independent random teacher
functions it discovers useful specialization at full capacity, but the capacity
curve is not monotonic at intermediate page counts. Do not treat it as a passed
routing algorithm.
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
        self.max_bits = max_bits
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


def top2_pages_and_scores(
    logits: torch.Tensor, used_bits: int, max_bits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = logits.shape[0]
    if used_bits == 0:
        page = torch.zeros(batch, dtype=torch.long, device=logits.device)
        return page[:, None], torch.zeros(batch, 1, device=logits.device)

    hard = logits > 0
    best_page = bits_to_page(hard, used_bits)

    # Under the factorized Bernoulli score, the runner-up page flips the active
    # factor with the smallest absolute logit margin.
    margins = logits.abs().clone()
    if used_bits < max_bits:
        margins[:, used_bits:] = float("inf")
    flip_index = margins.argmin(dim=1)
    second_bits = hard.clone()
    second_bits[torch.arange(batch), flip_index] = ~second_bits[
        torch.arange(batch), flip_index
    ]
    second_page = bits_to_page(second_bits, used_bits)

    active_logits = logits[:, :used_bits]
    best_active = hard[:, :used_bits]
    second_active = second_bits[:, :used_bits]
    best_score = -nn.functional.softplus(
        -torch.where(best_active, active_logits, -active_logits)
    ).sum(dim=1)
    second_score = -nn.functional.softplus(
        -torch.where(second_active, active_logits, -active_logits)
    ).sum(dim=1)

    return (
        torch.stack((best_page, second_page), dim=1),
        torch.stack((best_score, second_score), dim=1),
    )


def factor_balance_loss(logits: torch.Tensor) -> torch.Tensor:
    """Encourage balanced and weakly correlated binary factors without O(N) state."""
    probabilities = torch.sigmoid(logits)
    means = probabilities.mean(dim=0)
    marginal = torch.mean((means - 0.5) ** 2)
    centered = probabilities - means
    covariance = centered.T @ centered / probabilities.shape[0]
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return marginal + torch.mean(off_diagonal**2)


def normalized_entropy(counts: np.ndarray) -> float:
    if len(counts) <= 1:
        return 1.0
    positive = counts[counts > 0].astype(np.float64)
    probabilities = positive / positive.sum()
    return float(
        -(probabilities * np.log(probabilities)).sum() / np.log(len(counts))
    )


def run_one(args: argparse.Namespace, used_bits: int) -> dict[str, float]:
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.num_items != 2**args.max_bits:
        raise ValueError("G2d currently uses num_items == 2^max_bits")

    # Dense random prototypes deliberately remove any hand-crafted address
    # semantics. Teacher functions are independent across semantic items.
    base_context = rng.normal(size=(args.num_items, args.context_dim)).astype(np.float32)
    base_context /= np.linalg.norm(base_context, axis=1, keepdims=True) + 1e-12
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

    context_train = torch.from_numpy(train_context)
    context_val = torch.from_numpy(val_context)
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
        raise ValueError("page MLP does not fit inside block_bytes")

    optimizer = torch.optim.Adam(
        [
            {"params": router.parameters(), "lr": args.router_lr},
            {"params": pages.parameters(), "lr": args.page_lr},
        ]
    )

    for _ in range(args.epochs):
        order = torch.randperm(len(train_items))
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            logits = router(context_train[idx])
            candidate_pages, scores = top2_pages_and_scores(
                logits, used_bits, args.max_bits
            )
            if candidate_pages.shape[1] == 1:
                prediction = pages(candidate_pages[:, 0], x_train[idx])
            else:
                first = pages(candidate_pages[:, 0], x_train[idx])
                second = pages(candidate_pages[:, 1], x_train[idx])
                mixture = torch.softmax(scores / args.temperature, dim=1)
                prediction = (
                    mixture[:, 0, None] * first
                    + mixture[:, 1, None] * second
                )
            task_loss = torch.mean((prediction - y_train[idx]) ** 2)
            balance = factor_balance_loss(logits)
            loss = task_loss + args.balance_weight * balance
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    router.eval()
    pages.eval()
    with torch.no_grad():
        logits = router(context_val)
        hard_bits = logits > 0
        selected_pages = bits_to_page(hard_bits, used_bits)
        prediction = pages(selected_pages, x_val)
        normalized_mse = float(
            torch.mean((prediction - y_val) ** 2) / torch.mean(y_val**2)
        )
        counts = torch.bincount(
            selected_pages.cpu(), minlength=num_pages
        ).numpy()

    routed = selected_pages.cpu().numpy().reshape(
        args.num_items, args.val_queries_per_item
    )
    modal_pages: list[int] = []
    stability: list[float] = []
    for row in routed:
        local = np.bincount(row, minlength=num_pages)
        modal_pages.append(int(local.argmax()))
        stability.append(float(local.max() / len(row)))

    return {
        "used_bits": float(used_bits),
        "pages": float(num_pages),
        "normalized_mse": normalized_mse,
        "utilization_entropy": normalized_entropy(counts),
        "dead_page_fraction": float(np.mean(counts == 0)),
        "routing_stability": float(np.mean(stability)),
        "modal_item_collision_fraction": float(
            1.0 - len(set(modal_pages)) / args.num_items
        ),
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

    print("# ParamProbe G2d: task-only top-2 train / top-1 inference")
    print(f"seed={args.seed}")
    print(f"controller_params={controller_params}")
    print(f"routing_macs_per_query={routing_macs}")
    print(f"active_operator_macs_per_inference={operator_macs}")
    print("q_train=2_when_pages_gt_1")
    print("q_inference=1")
    print(f"block_bytes={args.block_bytes}")
    print(f"logical_bytes_per_inference={args.block_bytes}")
    print("router_supervision=task_loss_only")
    print()
    print(
        "used_bits\tpages\texternal_MiB\tnmse\tutil_entropy\tdead_pages\t"
        "route_stability\tmodal_item_collision"
    )

    sweep = [0] + list(range(args.min_bits, args.max_bits + 1, args.bit_step))
    if sweep[-1] != args.max_bits:
        sweep.append(args.max_bits)
    for used_bits in sweep:
        result = run_one(args, used_bits)
        external_mib = result["pages"] * args.block_bytes / (1024**2)
        print(
            f"{used_bits}\t{int(result['pages'])}\t{external_mib:.6f}\t"
            f"{result['normalized_mse']:.8f}\t{result['utilization_entropy']:.8f}\t"
            f"{result['dead_page_fraction']:.8f}\t{result['routing_stability']:.8f}\t"
            f"{result['modal_item_collision_fraction']:.8f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-items", type=int, default=64)
    parser.add_argument("--max-bits", type=int, default=6)
    parser.add_argument("--min-bits", type=int, default=2)
    parser.add_argument("--bit-step", type=int, default=2)
    parser.add_argument("--context-dim", type=int, default=32)
    parser.add_argument("--router-hidden-dim", type=int, default=64)
    parser.add_argument("--query-dim", type=int, default=8)
    parser.add_argument("--page-hidden-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--train-queries-per-item", type=int, default=64)
    parser.add_argument("--val-queries-per-item", type=int, default=24)
    parser.add_argument("--context-noise", type=float, default=0.08)
    parser.add_argument("--item-nuisance", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--router-lr", type=float, default=3e-3)
    parser.add_argument("--page-lr", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--balance-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    run(parser.parse_args())
