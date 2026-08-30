from pathlib import Path

import numpy as np

from paramprobe.page_mlp import (
    PageMLPLayout,
    apply_page_mlp,
    apply_page_mlp_block,
    encode_page_mlp,
)
from paramprobe.store import FileParameterStore


def test_page_mlp_fits_one_block_and_matches_resident(tmp_path: Path):
    rng = np.random.default_rng(101)
    layout = PageMLPLayout(input_dim=8, hidden_dim=32, output_dim=8)
    block_bytes = 4096
    assert layout.payload_bytes == 2208
    assert layout.payload_bytes <= block_bytes

    w1 = rng.normal(size=(layout.hidden_dim, layout.input_dim)).astype(np.float32)
    b1 = rng.normal(size=layout.hidden_dim).astype(np.float32)
    w2 = rng.normal(size=(layout.output_dim, layout.hidden_dim)).astype(np.float32)
    b2 = rng.normal(size=layout.output_dim).astype(np.float32)
    block = encode_page_mlp(w1, b1, w2, b2, block_bytes)
    x = rng.normal(size=layout.input_dim).astype(np.float32)

    expected = w2 @ np.tanh(w1 @ x + b1) + b2
    np.testing.assert_array_equal(apply_page_mlp_block(block, x, layout), expected)

    path = tmp_path / "page_mlp.bin"
    path.write_bytes(block)
    with FileParameterStore(path, block_bytes) as store:
        got = apply_page_mlp(store, 0, x, layout)
        assert store.stats.probes == 1
        assert store.stats.bytes_read == block_bytes
    np.testing.assert_array_equal(got, expected)
