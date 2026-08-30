from pathlib import Path

import numpy as np

from paramprobe.operator import apply_selected, encode_rank1
from paramprobe.store import FileParameterStore, InMemoryParameterStore


def test_file_store_probe_accounting_and_exactness(tmp_path: Path):
    rng = np.random.default_rng(11)
    d = 32
    block_bytes = 4096
    blocks = [
        encode_rank1(rng.normal(size=d), rng.normal(size=d), block_bytes)
        for _ in range(16)
    ]
    path = tmp_path / "params.bin"
    path.write_bytes(b"".join(blocks))

    x = rng.normal(size=d).astype(np.float32)
    selected = [2, 9, 13]

    memory_store = InMemoryParameterStore(blocks)
    expected = apply_selected(memory_store, selected, x)

    with FileParameterStore(path, block_bytes) as file_store:
        got = apply_selected(file_store, selected, x)
        assert file_store.stats.probes == len(selected)
        assert file_store.stats.bytes_read == len(selected) * block_bytes

    np.testing.assert_array_equal(got, expected)
