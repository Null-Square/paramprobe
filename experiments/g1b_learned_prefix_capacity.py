"""G1b: learned routing with a fixed controller across a capacity sweep.

The key methodological constraint is that the router is trained exactly once for
`max_bits` binary address factors. Every capacity point reuses the exact same
router, parameters, and routing compute. A smaller external memory simply uses a
shorter prefix of the same predicted address.

This experiment deliberately exposes the address-reliability problem of raw
factorized routing: if each factor is correct with probability p, full-address
accuracy behaves roughly like p**r when factor errors are weakly dependent.
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
    ids = np.zeros(bits.shape[0], dtype=np.int64)
    for j in range(used_bits):
        ids = (ids << 1) | bits[:, j].astype(np.int64)
    return ids


def fit_page_values(
    ids: np.ndarray, targets: np.ndarray, num_pages: int
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((num_pages, targets.shape[1]), dtype=np.float64)
    counts = np.bincount(ids, minlength=num_pages).astype(np.int64)
    np.add.at(sums, ids, targets)
    occupied = counts > 0
    sums[occupied] /= counts[occupied, None]
    return sums.astype(np.float32), counts


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


def make_samples(
    base: np.ndarray,
    bits: np.ndarray,
    targets: np.ndarray,
    repeats: int,
    noise: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.repeat(base, repeats, axis=0)
    x = x + rng.normal(scale=noise, size=x.shape).astype(np.float32)
    b = np.repeat(bits, repeats, axis=0)
    y = np.repeat(targets, repeats, axis=0)
    return x, b, y


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.num_items > 2**args.max_bits:
        raise ValueError("num_items cannot exceed the full address space")

    addresses = rng.choice(
        2**args.max_bits, size=args.num_items, replace=False
    ).astype(np.int64)
    bits = ints_to_bits(addresses, args.max_bits)
    signs = bits * 2.0 - 1.0

    encoder = rng.normal(size=(args.max_bits, args.input_dim)).astype(np.float32)
    encoder /= np.linalg.norm(encoder, axis=1, keepdims=True) + 1e-12
    base = (signs @ encoder) / math.sqrt(args.max_bits)
    # Fixed per-item nuisance prevents the task from being a trivial linear inverse.
    base += rng.normal(
        scale=args.item_nuisance,
        size=(args.num_items, args.input_dim),
    ).astype(np.float32)

    targets = rng.normal(size=(args.num_items, args.target_dim)).astype(np.float32)
    targets -= targets.mean(axis=0, keepdims=True)
    targets /= targets.std(axis=0, keepdims=True) + 1e-8

    x_train, bits_train, y_train = make_samples(
        base, bits, targets, args.train_repeats, args.noise, rng
    )
    x_val, bits_val, y_val = make_samples(
        base, bits, targets, args.val_repeats, args.noise, rng
    )

    model = FixedRouter(args.input_dim, args.hidden_dim, args.max_bits)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    x_train_t = torch.from_numpy(x_train)
    bits_train_t = torch.from_numpy(bits_train)
    n = len(x_train_t)

    for epoch in range(args.epochs):
        order = torch.randperm(n)
        model.train()
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            logits = model(x_train_t[idx])
            loss = loss_fn(logits, bits_train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_train = (model(torch.from_numpy(x_train)) > 0).numpy().astype(np.int8)
        pred_val = (model(torch.from_numpy(x_val)) > 0).numpy().astype(np.int8)

    controller_params = sum(p.numel() for p in model.parameters())
    controller_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    routing_macs = (
        args.input_dim * args.hidden_dim + args.hidden_dim * args.max_bits
    )
    bit_accuracy = float((pred_val == bits_val).mean())
    full_accuracy = float(np.all(pred_val == bits_val, axis=1).mean())
    independence_prediction = bit_accuracy**args.max_bits

    print("# ParamProbe G1b: fixed-controller learned routing capacity")
    print(f"seed={args.seed}")
    print(f"num_items={args.num_items}")
    print(f"max_bits={args.max_bits}")
    print(f"controller_params={controller_params}")
    print(f"controller_bytes={controller_bytes}")
    print(f"routing_macs_per_query={routing_macs}")
    print("q=1")
    print(f"block_bytes={args.block_bytes}")
    print(f"logical_bytes_per_query={args.block_bytes}")
    print(f"val_bit_accuracy={bit_accuracy:.8f}")
    print(f"val_full_address_accuracy={full_accuracy:.8f}")
    print(f"independent_factor_prediction={independence_prediction:.8f}")
    print(
        "used_bits\tpages\texternal_MiB\tprefix_acc\toccupied_train\t"
        "learned_mse\toracle_mse"
    )

    for used_bits in range(args.min_bits, args.max_bits + 1):
        num_pages = 1 << used_bits
        train_pred_ids = prefix_ids(pred_train, used_bits)
        val_pred_ids = prefix_ids(pred_val, used_bits)
        train_true_ids = prefix_ids(bits_train.astype(np.int8), used_bits)
        val_true_ids = prefix_ids(bits_val.astype(np.int8), used_bits)

        learned_values, counts = fit_page_values(
            train_pred_ids, y_train, num_pages
        )
        learned_prediction = learned_values[val_pred_ids]
        learned_mse = float(np.mean((y_val - learned_prediction) ** 2))

        oracle_values, _ = fit_page_values(train_true_ids, y_train, num_pages)
        oracle_prediction = oracle_values[val_true_ids]
        oracle_mse = float(np.mean((y_val - oracle_prediction) ** 2))

        prefix_accuracy = float(
            np.all(pred_val[:, :used_bits] == bits_val[:, :used_bits], axis=1).mean()
        )
        occupied = int(np.count_nonzero(counts))
        external_mib = num_pages * args.block_bytes / (1024**2)

        print(
            f"{used_bits}\t{num_pages}\t{external_mib:.6f}\t"
            f"{prefix_accuracy:.8f}\t{occupied}\t{learned_mse:.8f}\t"
            f"{oracle_mse:.8f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-items", type=int, default=4096)
    parser.add_argument("--max-bits", type=int, default=12)
    parser.add_argument("--min-bits", type=int, default=6)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--target-dim", type=int, default=16)
    parser.add_argument("--train-repeats", type=int, default=2)
    parser.add_argument("--val-repeats", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.12)
    parser.add_argument("--item-nuisance", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=41)
    run(parser.parse_args())
