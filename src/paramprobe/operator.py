"""Tiny page-sized neural operator encoding used by the G0 exactness test."""

from __future__ import annotations

import numpy as np


def encoded_rank1_bytes(d: int, dtype=np.float32) -> int:
    return 2 * d * np.dtype(dtype).itemsize


def encode_rank1(u: np.ndarray, v: np.ndarray, block_bytes: int) -> bytes:
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    if u.ndim != 1 or v.shape != u.shape:
        raise ValueError("u and v must be same-shaped vectors")
    payload = u.tobytes(order="C") + v.tobytes(order="C")
    if len(payload) > block_bytes:
        raise ValueError("operator does not fit in block")
    return payload + bytes(block_bytes - len(payload))


def apply_rank1_block(block: bytes, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    d = x.size
    need = encoded_rank1_bytes(d)
    if len(block) < need:
        raise ValueError("block too small for input dimension")
    values = np.frombuffer(block[:need], dtype=np.float32)
    u = values[:d]
    v = values[d : 2 * d]
    return np.float32(np.dot(u, x)) * v


def apply_selected(store, block_ids: list[int], x: np.ndarray) -> np.ndarray:
    """Stream selected blocks one at a time into a fixed-size accumulator."""
    y = np.asarray(x, dtype=np.float32).copy()
    for block_id in block_ids:
        y += apply_rank1_block(store.read_block(block_id), x)
    return y
