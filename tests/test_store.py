from pathlib import Path

import numpy as np
import pytest

from paramprobe.operator import apply_selected, encode_rank1
from paramprobe.store import (
    DirectIOParameterStore,
    FileParameterStore,
    InMemoryParameterStore,
)


def _fixture_blocks() -> tuple[np.random.Generator, int, list[bytes]]:
    rng = np.random.default_rng(11)
    d = 32
    block_bytes = 4096
    blocks = [
        encode_rank1(rng.normal(size=d), rng.normal(size=d), block_bytes)
        for _ in range(16)
    ]
    return rng, block_bytes, blocks


def test_file_store_probe_accounting_and_exactness(tmp_path: Path):
    rng, block_bytes, blocks = _fixture_blocks()
    path = tmp_path / "params.bin"
    path.write_bytes(b"".join(blocks))

    x = rng.normal(size=32).astype(np.float32)
    selected = [2, 9, 13]

    memory_store = InMemoryParameterStore(blocks)
    expected = apply_selected(memory_store, selected, x)

    with FileParameterStore(path, block_bytes) as file_store:
        got = apply_selected(file_store, selected, x)
        assert file_store.stats.probes == len(selected)
        assert file_store.stats.bytes_read == len(selected) * block_bytes

    np.testing.assert_array_equal(got, expected)


def test_direct_io_store_probe_accounting_and_exactness(tmp_path: Path):
    rng, block_bytes, blocks = _fixture_blocks()
    path = tmp_path / "params_direct.bin"
    path.write_bytes(b"".join(blocks))

    x = rng.normal(size=32).astype(np.float32)
    selected = [1, 7, 15]
    expected = apply_selected(InMemoryParameterStore(blocks), selected, x)

    try:
        store = DirectIOParameterStore(path, block_bytes)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"direct I/O unsupported on this host/filesystem: {exc}")

    with store:
        got = apply_selected(store, selected, x)
        assert store.stats.probes == len(selected)
        assert store.stats.bytes_read == len(selected) * block_bytes

    np.testing.assert_array_equal(got, expected)
