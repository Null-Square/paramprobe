"""G1c: error-correcting parameter addresses under a matched router budget.

A raw 11-bit router is compared with a systematic Hamming(15,11)-coded router.
Both receive the same input, activate one page, read the same number of external
bytes, and use nearly identical controller parameter/MAC budgets. The coded
router predicts 15 code bits and syndrome-decodes them back to an 11-bit page
address before the external parameter probe.

Hamming(15,11) is a finite proof-of-concept, not the asymptotic construction we
would use in a theorem. An asymptotically good positive-rate code family is the
appropriate theoretical object; see docs/theory_address_reliability.md.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

DATA_POSITIONS = (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
PARITY_POSITIONS = (1, 2, 4, 8)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ints_to_bits(values: np.ndarray, width: int) -> np.ndarray:
    shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
    return ((values[:, None] >> shifts) & 1).astype(np.int8)


def hamming1511_encode(data_bits: np.ndarray) -> np.ndarray:
    if data_bits.ndim != 2 or data_bits.shape[1] != 11:
        raise ValueError("Hamming(15,11) expects shape [n, 11]")
    out = np.zeros((len(data_bits), 15), dtype=np.int8)
    for j, pos in enumerate(DATA_POSITIONS):
        out[:, pos - 1] = data_bits[:, j]
    for parity_pos in PARITY_POSITIONS:
        covered = [
            pos
            for pos in range(1, 16)
            if (pos & parity_pos) and pos != parity_pos
        ]
        out[:, parity_pos - 1] = np.bitwise_xor.reduce(
            out[:, np.asarray(covered) - 1], axis=1
        )
    return out


def hamming1511_decode(code_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if code_bits.ndim != 2 or code_bits.shape[1] != 15:
        raise ValueError("Hamming(15,11) expects shape [n, 15]")
    corrected = code_bits.copy().astype(np.int8)
    syndrome = np.zeros(len(corrected), dtype=np.int64)
    for parity_pos in PARITY_POSITIONS:
        covered = [pos for pos in range(1, 16) if pos & parity_pos]
        parity = np.bitwise_xor.reduce(
            corrected[:, np.asarray(covered) - 1], axis=1
        )
        syndrome += parity.astype(np.int64) * parity_pos
    rows = np.nonzero(syndrome)[0]
    valid = rows[(syndrome[rows] >= 1) & (syndrome[rows] <= 15)]
    corrected[valid, syndrome[valid] - 1] ^= 1
    data = corrected[:, np.asarray(DATA_POSITIONS) - 1]
    return data, syndrome


def prefix_ids(bits: np.ndarray, used_bits: int) -> np.ndarray:
    ids = np.zeros(bits.shape[0], dtype=np.int64)
    for j in range(used_bits):
        ids = (ids << 1) | bits[:, j].astype(np.int64)
    return ids


def fit_page_values(ids: np.ndarray, targets: np.ndarray, pages: int) -> np.ndarray:
    sums = np.zeros((pages, targets.shape[1]), dtype=np.float64)
    counts = np.bincount(ids, minlength=pages).astype(np.int64)
    np.add.at(sums, ids, targets)
    occupied = counts > 0
    sums[occupied] /= counts[occupied, None]
    return sums.astype(np.float32)


class Router(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_bits: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_bits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TrainedRouter:
    train_bits: np.ndarray
    val_bits: np.ndarray
    params: int
    macs: int
    output_bit_accuracy: float


def train_router(
    x_train: np.ndarray,
    y_train_bits: np.ndarray,
    x_val: np.ndarray,
    y_val_bits: np.ndarray,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> TrainedRouter:
    output_bits = y_train_bits.shape[1]
    model = Router(x_train.shape[1], hidden_dim, output_bits)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train_bits.astype(np.float32))
    n = len(x_train_t)

    for _ in range(epochs):
        order = torch.randperm(n)
        model.train()
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            logits = model(x_train_t[idx])
            loss = loss_fn(logits, y_train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_bits = (model(x_train_t) > 0).numpy().astype(np.int8)
        val_bits = (model(torch.from_numpy(x_val)) > 0).numpy().astype(np.int8)

    params = sum(p.numel() for p in model.parameters())
    macs = x_train.shape[1] * hidden_dim + hidden_dim * output_bits
    output_bit_accuracy = float((val_bits == y_val_bits).mean())
    return TrainedRouter(train_bits, val_bits, params, macs, output_bit_accuracy)


def run_seed(args: argparse.Namespace, seed: int) -> tuple[float, float, float, float]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    logical_bits = 11
    num_items = 2**logical_bits

    addresses = rng.permutation(num_items).astype(np.int64)
    data_bits = ints_to_bits(addresses, logical_bits)
    code_bits = hamming1511_encode(data_bits)

    # Both routers receive exactly the same redundant input representation.
    signs = code_bits.astype(np.float32) * 2.0 - 1.0
    encoder = rng.normal(size=(15, args.input_dim)).astype(np.float32)
    encoder /= np.linalg.norm(encoder, axis=1, keepdims=True) + 1e-12
    base = (signs @ encoder) / math.sqrt(15)
    base += rng.normal(
        scale=args.item_nuisance, size=(num_items, args.input_dim)
    ).astype(np.float32)

    targets = rng.normal(size=(num_items, args.target_dim)).astype(np.float32)
    targets -= targets.mean(axis=0, keepdims=True)
    targets /= targets.std(axis=0, keepdims=True) + 1e-8

    def samples(repeats: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.repeat(base, repeats, axis=0)
        x += rng.normal(scale=args.noise, size=x.shape).astype(np.float32)
        d = np.repeat(data_bits, repeats, axis=0)
        c = np.repeat(code_bits, repeats, axis=0)
        y = np.repeat(targets, repeats, axis=0)
        return x, d, c, y

    x_train, data_train, code_train, y_train = samples(args.train_repeats)
    x_val, data_val, code_val, y_val = samples(args.val_repeats)

    raw = train_router(
        x_train,
        data_train,
        x_val,
        data_val,
        args.raw_hidden,
        args.epochs,
        args.batch_size,
        args.lr,
    )
    coded = train_router(
        x_train,
        code_train,
        x_val,
        code_val,
        args.coded_hidden,
        args.epochs,
        args.batch_size,
        args.lr,
    )

    coded_train_data, _ = hamming1511_decode(coded.train_bits)
    coded_val_data, _ = hamming1511_decode(coded.val_bits)

    raw_full = float(np.all(raw.val_bits == data_val, axis=1).mean())
    coded_decoded_bit = float((coded_val_data == data_val).mean())
    coded_full = float(np.all(coded_val_data == data_val, axis=1).mean())

    print(f"# seed={seed}")
    print(
        f"raw_params={raw.params} raw_macs={raw.macs} "
        f"coded_params={coded.params} coded_macs={coded.macs}"
    )
    print(
        f"raw_output_bit_acc={raw.output_bit_accuracy:.8f} "
        f"raw_full_addr_acc={raw_full:.8f}"
    )
    print(
        f"coded_output_bit_acc={coded.output_bit_accuracy:.8f} "
        f"coded_decoded_bit_acc={coded_decoded_bit:.8f} "
        f"coded_full_addr_acc={coded_full:.8f}"
    )
    print("used_bits\tpages\texternal_MiB\traw_mse\tcoded_mse\toracle_mse")

    raw_full_mse = float("nan")
    coded_full_mse = float("nan")
    for used_bits in range(args.min_bits, logical_bits + 1):
        pages = 1 << used_bits
        true_train_ids = prefix_ids(data_train, used_bits)
        true_val_ids = prefix_ids(data_val, used_bits)
        raw_train_ids = prefix_ids(raw.train_bits, used_bits)
        raw_val_ids = prefix_ids(raw.val_bits, used_bits)
        coded_train_ids = prefix_ids(coded_train_data, used_bits)
        coded_val_ids = prefix_ids(coded_val_data, used_bits)

        oracle_values = fit_page_values(true_train_ids, y_train, pages)
        raw_values = fit_page_values(raw_train_ids, y_train, pages)
        coded_values = fit_page_values(coded_train_ids, y_train, pages)

        oracle_mse = float(np.mean((y_val - oracle_values[true_val_ids]) ** 2))
        raw_mse = float(np.mean((y_val - raw_values[raw_val_ids]) ** 2))
        coded_mse = float(np.mean((y_val - coded_values[coded_val_ids]) ** 2))
        if used_bits == logical_bits:
            raw_full_mse = raw_mse
            coded_full_mse = coded_mse

        external_mib = pages * args.block_bytes / (1024**2)
        print(
            f"{used_bits}\t{pages}\t{external_mib:.6f}\t{raw_mse:.8f}\t"
            f"{coded_mse:.8f}\t{oracle_mse:.8f}"
        )
    return raw_full, coded_full, raw_full_mse, coded_full_mse


def run(args: argparse.Namespace) -> None:
    print("# ParamProbe G1c: coded parameter addressing")
    print("code=Hamming(15,11)")
    print("q=1")
    print(f"block_bytes={args.block_bytes}")
    print(f"logical_bytes_per_query={args.block_bytes}")
    print(f"noise={args.noise}")

    rows = [run_seed(args, seed) for seed in args.seeds]
    values = np.asarray(rows, dtype=np.float64)
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros(4)
    names = ("raw_full_acc", "coded_full_acc", "raw_full_mse", "coded_full_mse")
    print("# aggregate")
    for i, name in enumerate(names):
        print(f"{name}_mean={means[i]:.8f} {name}_std={stds[i]:.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 19])
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--raw-hidden", type=int, default=123)
    parser.add_argument("--coded-hidden", type=int, default=117)
    parser.add_argument("--target-dim", type=int, default=16)
    parser.add_argument("--train-repeats", type=int, default=2)
    parser.add_argument("--val-repeats", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--item-nuisance", type=float, default=0.12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--min-bits", type=int, default=5)
    run(parser.parse_args())
