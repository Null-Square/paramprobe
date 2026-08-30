"""G0: deterministic probe-budget and exactness smoke experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np

from paramprobe.operator import apply_selected, encode_rank1
from paramprobe.store import FileParameterStore, InMemoryParameterStore


def run(num_blocks: int, block_bytes: int, d: int, q: int, trials: int, seed: int) -> None:
    if q > num_blocks:
        raise ValueError("q must not exceed num_blocks")
    rng = np.random.default_rng(seed)

    with tempfile.TemporaryDirectory(prefix="paramprobe-g0-") as tmp:
        path = Path(tmp) / "params.bin"
        blocks: list[bytes] = []
        with path.open("wb") as f:
            for _ in range(num_blocks):
                block = encode_rank1(
                    rng.normal(size=d), rng.normal(size=d), block_bytes
                )
                blocks.append(block)
                f.write(block)

        memory_store = InMemoryParameterStore(blocks)
        max_abs_diff = 0.0
        with FileParameterStore(path, block_bytes) as file_store:
            for _ in range(trials):
                x = rng.normal(size=d).astype(np.float32)
                selected = rng.choice(num_blocks, size=q, replace=False).tolist()
                memory_store.stats.reset()
                file_store.stats.reset()
                expected = apply_selected(memory_store, selected, x)
                got = apply_selected(file_store, selected, x)
                max_abs_diff = max(max_abs_diff, float(np.max(np.abs(got - expected))))
                assert file_store.stats.probes == q
                assert file_store.stats.bytes_read == q * block_bytes

        print(f"num_blocks={num_blocks}")
        print(f"external_capacity_bytes={num_blocks * block_bytes}")
        print(f"block_bytes={block_bytes}")
        print(f"q={q}")
        print(f"explicit_bytes_per_invocation={q * block_bytes}")
        print(f"trials={trials}")
        print(f"max_abs_diff={max_abs_diff:.9g}")
        print("status=PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-blocks", type=int, default=4096)
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    run(args.num_blocks, args.block_bytes, args.d, args.q, args.trials, args.seed)
