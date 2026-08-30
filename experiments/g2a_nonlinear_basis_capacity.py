"""G2a: nonlinear operator capacity under a fixed one-page probe budget.

Each semantic item i owns an external matrix A_i. The evaluated function is

    y = A_i tanh(R z + b),

where the random nonlinear feature map (R, b) is resident and fixed. A physical
parameter page stores exactly one A_p matrix (64 x 16 FP32 = 4096 bytes).

When several semantic items collide on one page, the population-risk minimizer
is the mean of their A_i matrices. This makes the nonlinear-operator collision
floor analytically tractable while still requiring a real nonlinear function
of z at inference time.

The router is trained once at maximum address width. Every capacity point reuses
the exact same router and compute graph; smaller memories use shorter prefixes.
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


def prefix_ids(bits: np.ndarray, used_bits: int) -> np.ndarray:
    if used_bits == 0:
        return np.zeros(bits.shape[0], dtype=np.int64)
    ids = np.zeros(bits.shape[0], dtype=np.int64)
    for j in range(used_bits):
        ids = (ids << 1) | bits[:, j].astype(np.int64)
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


def fit_mean_pages(
    ids: np.ndarray, teacher_weights: np.ndarray, num_pages: int
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(
        (num_pages, teacher_weights.shape[1], teacher_weights.shape[2]),
        dtype=np.float64,
    )
    counts = np.bincount(ids, minlength=num_pages).astype(np.int64)
    np.add.at(sums, ids, teacher_weights)
    occupied = counts > 0
    sums[occupied] /= counts[occupied, None, None]
    return sums.astype(np.float32), counts


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.num_items > 2**args.max_bits:
        raise ValueError("num_items cannot exceed full address space")
    if args.feature_dim * args.output_dim * 4 != args.block_bytes:
        raise ValueError(
            "G2a expects one FP32 output matrix to fill exactly one block: "
            "feature_dim * output_dim * 4 == block_bytes"
        )

    addresses = rng.choice(
        2**args.max_bits, size=args.num_items, replace=False
    ).astype(np.int64)
    bits = ints_to_bits(addresses, args.max_bits)
    signs = bits * 2.0 - 1.0

    encoder = rng.normal(size=(args.max_bits, args.input_dim)).astype(np.float32)
    encoder /= np.linalg.norm(encoder, axis=1, keepdims=True) + 1e-12
    base_context = (signs @ encoder) / math.sqrt(args.max_bits)
    base_context += rng.normal(
        scale=args.item_nuisance,
        size=base_context.shape,
    ).astype(np.float32)

    x_train = np.repeat(base_context, args.router_train_repeats, axis=0)
    x_train += rng.normal(
        scale=args.context_noise, size=x_train.shape
    ).astype(np.float32)
    bits_train = np.repeat(bits, args.router_train_repeats, axis=0)
    x_val = base_context + rng.normal(
        scale=args.context_noise, size=base_context.shape
    ).astype(np.float32)

    router = FixedRouter(args.input_dim, args.router_hidden_dim, args.max_bits)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=1e-4)
    x_train_t = torch.from_numpy(x_train)
    bits_train_t = torch.from_numpy(bits_train)

    for _ in range(args.epochs):
        order = torch.randperm(len(x_train_t))
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            logits = router(x_train_t[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, bits_train_t[idx]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    router.eval()
    with torch.no_grad():
        # A canonical route is used to fit external pages. Validation uses an
        # independent noisy context to expose routing generalization failures.
        pred_train_bits = (router(torch.from_numpy(base_context)) > 0).numpy().astype(np.int8)
        pred_val_bits = (router(torch.from_numpy(x_val)) > 0).numpy().astype(np.int8)

    bit_accuracy = float((pred_val_bits == bits).mean())
    full_accuracy = float(np.all(pred_val_bits == bits, axis=1).mean())

    teacher_weights = rng.normal(
        scale=1.0 / math.sqrt(args.feature_dim),
        size=(args.num_items, args.output_dim, args.feature_dim),
    ).astype(np.float32)
    feature_matrix = rng.normal(
        scale=1.0 / math.sqrt(args.query_dim),
        size=(args.feature_dim, args.query_dim),
    ).astype(np.float32)
    feature_bias = rng.normal(scale=0.3, size=args.feature_dim).astype(np.float32)

    item_ids = np.repeat(np.arange(args.num_items), args.queries_per_item)
    z = rng.normal(size=(len(item_ids), args.query_dim)).astype(np.float32)
    phi = np.tanh(z @ feature_matrix.T + feature_bias)
    targets = np.einsum("nof,nf->no", teacher_weights[item_ids], phi)
    target_energy = float(np.mean(targets**2))

    controller_params = sum(p.numel() for p in router.parameters())
    routing_macs = (
        args.input_dim * args.router_hidden_dim
        + args.router_hidden_dim * args.max_bits
    )
    resident_feature_macs = args.query_dim * args.feature_dim
    external_operator_macs = args.feature_dim * args.output_dim

    print("# ParamProbe G2a: nonlinear basis operator capacity")
    print(f"seed={args.seed}")
    print(f"num_items={args.num_items}")
    print(f"controller_params={controller_params}")
    print(f"routing_macs_per_query={routing_macs}")
    print(f"resident_feature_macs_per_query={resident_feature_macs}")
    print(f"external_operator_macs_per_query={external_operator_macs}")
    print("q=1")
    print(f"block_bytes={args.block_bytes}")
    print(f"logical_bytes_per_query={args.block_bytes}")
    print(f"router_bit_accuracy={bit_accuracy:.8f}")
    print(f"router_full_address_accuracy={full_accuracy:.8f}")
    print()
    print(
        "used_bits\tpages\texternal_MiB\tprefix_acc\toccupied\t"
        "oracle_normalized_mse\tlearned_normalized_mse\tanalytic_oracle"
    )

    sweep = [0] + list(range(args.min_bits, args.max_bits + 1))
    for used_bits in sweep:
        num_pages = 1 if used_bits == 0 else 1 << used_bits
        true_ids = prefix_ids(bits.astype(np.int8), used_bits)
        train_pred_ids = prefix_ids(pred_train_bits, used_bits)
        val_pred_ids = prefix_ids(pred_val_bits, used_bits)

        oracle_pages, oracle_counts = fit_mean_pages(
            true_ids, teacher_weights, num_pages
        )
        learned_pages, _ = fit_mean_pages(
            train_pred_ids, teacher_weights, num_pages
        )

        oracle_pred = np.einsum(
            "nof,nf->no", oracle_pages[true_ids[item_ids]], phi
        )
        learned_pred = np.einsum(
            "nof,nf->no", learned_pages[val_pred_ids[item_ids]], phi
        )
        oracle_nmse = float(np.mean((targets - oracle_pred) ** 2) / target_energy)
        learned_nmse = float(np.mean((targets - learned_pred) ** 2) / target_energy)
        prefix_accuracy = (
            1.0
            if used_bits == 0
            else float(
                np.all(
                    pred_val_bits[:, :used_bits] == bits[:, :used_bits], axis=1
                ).mean()
            )
        )
        occupied = int(np.count_nonzero(oracle_counts))
        analytic_oracle = 1.0 - occupied / args.num_items
        external_mib = num_pages * args.block_bytes / (1024**2)

        print(
            f"{used_bits}\t{num_pages}\t{external_mib:.6f}\t"
            f"{prefix_accuracy:.8f}\t{occupied}\t{oracle_nmse:.8f}\t"
            f"{learned_nmse:.8f}\t{analytic_oracle:.8f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-items", type=int, default=4096)
    parser.add_argument("--max-bits", type=int, default=12)
    parser.add_argument("--min-bits", type=int, default=4)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--router-hidden-dim", type=int, default=128)
    parser.add_argument("--query-dim", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=16)
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--queries-per-item", type=int, default=4)
    parser.add_argument("--router-train-repeats", type=int, default=2)
    parser.add_argument("--context-noise", type=float, default=0.04)
    parser.add_argument("--item-nuisance", type=float, default=0.04)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=1)
    run(parser.parse_args())
