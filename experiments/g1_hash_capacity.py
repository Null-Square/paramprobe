"""G1a: collision-limited associative capacity under a fixed one-probe budget.

This is deliberately not yet a neural-routing experiment. It isolates the core
question: can useful learned state scale with external capacity while resident
routing state, active compute, and parameter probes stay fixed?

Each integer key is mapped by a fixed stateless SplitMix64 router to one page.
Each page stores the least-squares-optimal value (the mean target of keys that
collide on that page). With iid zero-mean unit-variance targets, the expected
per-coordinate MSE conditioned on occupancy is exactly 1 - occupied/A.
For uniform hashing, E[occupied] = N * (1 - (1 - 1/N)**A).
"""

from __future__ import annotations

import argparse

import numpy as np

MASK64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def page_ids(num_items: int, num_pages: int) -> np.ndarray:
    if num_pages < 1 or num_pages & (num_pages - 1):
        raise ValueError("num_pages must be a positive power of two")
    mask = num_pages - 1
    return np.fromiter(
        (splitmix64(i) & mask for i in range(num_items)),
        dtype=np.int64,
        count=num_items,
    )


def optimal_page_values(
    ids: np.ndarray, targets: np.ndarray, num_pages: int
) -> tuple[np.ndarray, int]:
    sums = np.zeros((num_pages, targets.shape[1]), dtype=np.float64)
    counts = np.bincount(ids, minlength=num_pages).astype(np.int64)
    np.add.at(sums, ids, targets)
    occupied_mask = counts > 0
    sums[occupied_mask] /= counts[occupied_mask, None]
    return sums, int(np.count_nonzero(occupied_mask))


def expected_uniform_hash_mse(num_items: int, num_pages: int) -> float:
    expected_occupied = num_pages * (
        1.0 - (1.0 - 1.0 / num_pages) ** num_items
    )
    return 1.0 - expected_occupied / num_items


def run(
    num_items: int,
    target_dim: int,
    min_log2_pages: int,
    max_log2_pages: int,
    block_bytes: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    targets = rng.normal(size=(num_items, target_dim))
    targets -= targets.mean(axis=0, keepdims=True)
    targets /= targets.std(axis=0, keepdims=True)

    print("# ParamProbe G1a: hashed associative capacity")
    print(f"num_items={num_items}")
    print(f"target_dim={target_dim}")
    print("q=1")
    print(f"block_bytes={block_bytes}")
    print(f"logical_bytes_per_query={block_bytes}")
    print("router_state_bytes=0")
    print("router=stateless_splitmix64")
    print()
    print(
        "pages\texternal_MiB\toccupied\tempirical_mse\t"
        "occupancy_mse\tuniform_hash_expectation"
    )

    for log2_pages in range(min_log2_pages, max_log2_pages + 1):
        num_pages = 1 << log2_pages
        ids = page_ids(num_items, num_pages)
        values, occupied = optimal_page_values(ids, targets, num_pages)
        predictions = values[ids]
        empirical_mse = float(np.mean((targets - predictions) ** 2))
        occupancy_mse = 1.0 - occupied / num_items
        uniform_expected = expected_uniform_hash_mse(num_items, num_pages)
        external_mib = num_pages * block_bytes / (1024**2)
        print(
            f"{num_pages}\t{external_mib:.6f}\t{occupied}\t"
            f"{empirical_mse:.8f}\t{occupancy_mse:.8f}\t{uniform_expected:.8f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-items", type=int, default=16384)
    parser.add_argument("--target-dim", type=int, default=32)
    parser.add_argument("--min-log2-pages", type=int, default=6)
    parser.add_argument("--max-log2-pages", type=int, default=14)
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    run(
        args.num_items,
        args.target_dim,
        args.min_log2_pages,
        args.max_log2_pages,
        args.block_bytes,
        args.seed,
    )
