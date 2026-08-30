"""Serializable page-sized nonlinear MLP operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PageMLPLayout:
    """Shape metadata for a two-layer tanh operator stored in one page."""

    input_dim: int
    hidden_dim: int
    output_dim: int

    @property
    def parameter_count(self) -> int:
        return (
            self.hidden_dim * self.input_dim
            + self.hidden_dim
            + self.output_dim * self.hidden_dim
            + self.output_dim
        )

    @property
    def payload_bytes(self) -> int:
        return self.parameter_count * np.dtype(np.float32).itemsize


def encode_page_mlp(
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    block_bytes: int,
) -> bytes:
    w1 = np.asarray(w1, dtype=np.float32)
    b1 = np.asarray(b1, dtype=np.float32)
    w2 = np.asarray(w2, dtype=np.float32)
    b2 = np.asarray(b2, dtype=np.float32)

    if w1.ndim != 2 or w2.ndim != 2:
        raise ValueError("w1 and w2 must be matrices")
    hidden_dim, input_dim = w1.shape
    output_dim, hidden_dim_2 = w2.shape
    if hidden_dim_2 != hidden_dim:
        raise ValueError("hidden dimensions do not match")
    if b1.shape != (hidden_dim,) or b2.shape != (output_dim,):
        raise ValueError("bias shapes do not match weight shapes")

    values = np.concatenate((w1.ravel(), b1, w2.ravel(), b2))
    payload = values.tobytes(order="C")
    if len(payload) > block_bytes:
        raise ValueError("MLP operator does not fit in one parameter block")
    return payload + bytes(block_bytes - len(payload))


def apply_page_mlp_block(
    block: bytes | memoryview,
    x: np.ndarray,
    layout: PageMLPLayout,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape != (layout.input_dim,):
        raise ValueError("input shape does not match layout")
    if len(block) < layout.payload_bytes:
        raise ValueError("block is too small for MLP layout")

    values = np.frombuffer(block[: layout.payload_bytes], dtype=np.float32)
    offset = 0

    n_w1 = layout.hidden_dim * layout.input_dim
    w1 = values[offset : offset + n_w1].reshape(
        layout.hidden_dim, layout.input_dim
    )
    offset += n_w1

    b1 = values[offset : offset + layout.hidden_dim]
    offset += layout.hidden_dim

    n_w2 = layout.output_dim * layout.hidden_dim
    w2 = values[offset : offset + n_w2].reshape(
        layout.output_dim, layout.hidden_dim
    )
    offset += n_w2

    b2 = values[offset : offset + layout.output_dim]
    hidden = np.tanh(w1 @ x + b1)
    return (w2 @ hidden + b2).astype(np.float32, copy=False)


def apply_page_mlp(store, block_id: int, x: np.ndarray, layout: PageMLPLayout) -> np.ndarray:
    """Evaluate exactly one external nonlinear operator using one block probe."""
    return apply_page_mlp_block(store.read_block(block_id), x, layout)
