"""G4 precheck: aligned direct-I/O page-size sweep.

This benchmark measures explicit O_DIRECT reads so ordinary OS page-cache hits do
not masquerade as external parameter I/O. Device/controller caches still exist;
results are machine-specific and are not model-performance claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import tempfile
import time

from paramprobe.store import DirectIOParameterStore

MIB = 1024 * 1024


def create_backing_file(path: Path, size_mib: int) -> None:
    target_bytes = size_mib * MIB
    if path.exists() and path.stat().st_size == target_bytes:
        return
    # Repeated data is fine: this benchmark studies block I/O, not compression.
    chunk = bytes((i * 131 + 17) & 0xFF for i in range(MIB))
    with path.open("wb", buffering=MIB) as handle:
        for _ in range(size_mib):
            handle.write(chunk)


def benchmark(
    path: Path,
    block_bytes: int,
    target_read_mib: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed + block_bytes)
    with DirectIOParameterStore(path, block_bytes) as store:
        reads = max(1, (target_read_mib * MIB) // block_bytes)
        ids = [rng.randrange(store.num_blocks) for _ in range(reads)]
        start = time.perf_counter()
        for block_id in ids:
            data = store.read_block(block_id)
            if len(data) != block_bytes:
                raise AssertionError("unexpected block length")
        elapsed = time.perf_counter() - start
        expected_bytes = reads * block_bytes
        if store.stats.probes != reads or store.stats.bytes_read != expected_bytes:
            raise AssertionError("probe accounting mismatch")

    mib_read = expected_bytes / MIB
    return {
        "reads": float(reads),
        "mib_read": mib_read,
        "elapsed_s": elapsed,
        "probes_s": reads / elapsed,
        "mib_s": mib_read / elapsed,
        "us_probe": elapsed * 1e6 / reads,
    }


def run(args: argparse.Namespace) -> None:
    if args.path:
        path = Path(args.path)
        create_backing_file(path, args.file_mib)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="paramprobe-direct-")
        path = Path(cleanup.name) / "params.bin"
        create_backing_file(path, args.file_mib)

    print("# ParamProbe G4 direct-I/O page-size sweep")
    print(f"path={path}")
    print(f"file_MiB={args.file_mib}")
    print(f"target_read_MiB_per_point={args.target_read_mib}")
    print("mode=O_DIRECT+preadv")
    print("note=OS_page_cache_bypassed_device_caches_not_controlled")
    print("block_bytes\treads\tprobes_s\tMiB_s\tus_probe")

    try:
        for block_bytes in args.block_bytes:
            if (args.file_mib * MIB) % block_bytes:
                raise ValueError("file size must be divisible by every block size")
            result = benchmark(path, block_bytes, args.target_read_mib, args.seed)
            print(
                f"{block_bytes}\t{int(result['reads'])}\t"
                f"{result['probes_s']:.2f}\t{result['mib_s']:.2f}\t"
                f"{result['us_probe']:.2f}"
            )
    finally:
        if cleanup is not None:
            cleanup.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--file-mib", type=int, default=128)
    parser.add_argument("--target-read-mib", type=int, default=32)
    parser.add_argument(
        "--block-bytes",
        type=int,
        nargs="+",
        default=[4096, 16384, 65536, 262144],
    )
    parser.add_argument("--seed", type=int, default=71)
    run(parser.parse_args())
