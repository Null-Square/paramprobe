"""Fixed-block static KV export. Post-training pruning and elementary quantization.

This module does not implement a new learning algorithm or GPTQ. Every call to
FileKV.residuals reads a full page per query; it does not cache decoded pages.
The operating system may cache file data. Global router state is accounted
separately by the caller. All computations after decode use FP32.
"""
from __future__ import annotations
from pathlib import Path
import os
import struct
import threading
import numpy as np

MAGIC = b"PPKV0001"
HEADER = struct.Struct("<8sHBBHHIffI")
CODECS = {"fp32": (0, np.dtype("<f4")), "fp16": (1, np.dtype("<f2")),
          "int8": (2, np.dtype("i1"))}


def integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def top_mass_indices(mass: np.ndarray, k: int) -> np.ndarray:
    mass = np.asarray(mass, dtype=np.float64)
    k = integer(k, "k")
    if mass.ndim != 2 or not 1 <= k <= mass.shape[1]:
        raise ValueError("require [blocks,slots] mass and 1 <= k <= slots")
    if not np.isfinite(mass).all() or (mass < 0).any():
        raise ValueError("mass must be finite and nonnegative")
    return np.argsort(-mass, axis=1, kind="stable")[:, :k]


def encode_block(keys: np.ndarray, values: np.ndarray, block_bytes: int,
                 codec: str = "fp32") -> bytes:
    block_bytes = integer(block_bytes, "block_bytes")
    keys = np.asarray(keys, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if keys.ndim != 2 or values.shape != keys.shape or min(keys.shape) < 1:
        raise ValueError("keys and values need equal nonempty [slots,dimension] shapes")
    if not np.isfinite(keys).all() or not np.isfinite(values).all():
        raise ValueError("parameters must be finite")
    if max(keys.shape) > 65535 or codec not in CODECS:
        raise ValueError("unsupported shape or codec")
    number, dtype = CODECS[codec]
    sk = sv = np.float32(1)
    if codec == "int8":
        sk = np.float32(np.max(np.abs(keys)) / 127) if keys.any() else np.float32(1)
        sv = np.float32(np.max(np.abs(values)) / 127) if values.any() else np.float32(1)
        if min(sk, sv) <= 0:
            raise ValueError("quantization scale underflow")
        ek = np.clip(np.rint(keys / sk), -127, 127).astype(dtype)
        ev = np.clip(np.rint(values / sv), -127, 127).astype(dtype)
    else:
        with np.errstate(over="ignore"):
            ek, ev = keys.astype(dtype), values.astype(dtype)
        if not np.isfinite(ek).all() or not np.isfinite(ev).all():
            raise ValueError("codec overflow")
    payload = ek.tobytes(order="C") + ev.tobytes(order="C")
    header = HEADER.pack(MAGIC, 1, number, 0, *keys.shape, len(payload), sk, sv, 0)
    if len(header) + len(payload) > block_bytes:
        raise ValueError("encoded block exceeds its byte allowance")
    return header + payload + bytes(block_bytes - len(header) - len(payload))


def decode_block(block: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(block) < HEADER.size:
        raise ValueError("truncated header")
    magic, version, code, flags, slots, dim, n, sk, sv, reserved = HEADER.unpack_from(block)
    if magic != MAGIC or version != 1 or code not in (0, 1, 2) or flags or reserved:
        raise ValueError("invalid block header")
    dtype = (np.dtype("<f4"), np.dtype("<f2"), np.dtype("i1"))[code]
    if not slots or not dim or n != 2 * slots * dim * dtype.itemsize:
        raise ValueError("invalid payload shape/length")
    if HEADER.size + n > len(block) or not np.isfinite([sk,sv]).all() or min(sk,sv) <= 0:
        raise ValueError("invalid payload/scales")
    a = np.frombuffer(block, dtype=dtype, count=2*slots*dim, offset=HEADER.size)
    a = a.astype(np.float32).reshape(2, slots, dim)
    if code == 2:
        a[0] *= sk
        a[1] *= sv
    elif sk != 1.0 or sv != 1.0:
        raise ValueError("non-quantized scales must equal one")
    if not np.isfinite(a).all():
        raise ValueError("nonfinite decoded parameters")
    return a[0], a[1]


def export_store(path: Path, keys: np.ndarray, values: np.ndarray,
                 indices: np.ndarray, block_bytes: int, codec: str) -> int:
    keys, values, indices = np.asarray(keys), np.asarray(values), np.asarray(indices)
    if keys.ndim != 3 or keys.shape != values.shape or indices.ndim != 2:
        raise ValueError("invalid table shapes")
    if indices.shape[0] != keys.shape[0] or not np.issubdtype(indices.dtype,np.integer):
        raise ValueError("invalid index shape or dtype")
    if (indices < 0).any() or (indices >= keys.shape[1]).any():
        raise ValueError("index outside table")
    if any(np.unique(row).size != row.size for row in indices):
        raise ValueError("duplicate retained index")
    # Exclusive creation prevents accidental overwrite of previous evidence.
    with Path(path).open("xb") as f:
        for i, selection in enumerate(indices):
            f.write(encode_block(keys[i,selection], values[i,selection], block_bytes,codec))
    return Path(path).stat().st_size


def residual(keys: np.ndarray, values: np.ndarray, query: np.ndarray,
             scale: float = 0.15) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    if query.shape != (keys.shape[1],) or not np.isfinite(query).all():
        raise ValueError("invalid query")
    scores = keys @ query
    if not np.isfinite(scores).all():
        raise ValueError("score overflow")
    p = np.exp(scores - scores.max())
    p /= p.sum()
    return np.float32(scale) * np.tanh(p @ values)


class FileKV:
    """One explicit pread per query, with no decoded-page cache.

    Each query invocation has a fixed one-read implementation. This is not a
    security boundary against callers opening the same file themselves.
    Counters count attempts and actual bytes, even on short reads.
    """
    def __init__(self, path: Path, block_bytes: int, dimension: int = 96):
        self.block_bytes = integer(block_bytes,"block_bytes")
        self.dimension = integer(dimension,"dimension")
        if self.block_bytes < HEADER.size or self.dimension <= 0:
            raise ValueError("invalid block size or dimension")
        self.fd = os.open(path,os.O_RDONLY)
        size = os.fstat(self.fd).st_size
        if size == 0 or size % self.block_bytes:
            os.close(self.fd); self.fd = None
            raise ValueError("invalid store size")
        self.blocks = size // self.block_bytes
        self.reads = self.returned_bytes = self.failed_reads = 0
        self.lock = threading.Lock()

    def close(self):
        with self.lock:
            if self.fd is not None:
                os.close(self.fd); self.fd = None

    def __enter__(self): return self
    def __exit__(self,*args): self.close()

    def residuals(self, ids: np.ndarray, queries: np.ndarray) -> np.ndarray:
        ids, queries = np.asarray(ids), np.asarray(queries,dtype=np.float32)
        if not np.issubdtype(ids.dtype,np.integer) or queries.shape != (*ids.shape,self.dimension):
            raise ValueError("invalid ids/query shapes")
        if not np.isfinite(queries).all():
            raise ValueError("nonfinite queries")
        if (ids < 0).any() or (ids >= self.blocks).any():
            raise IndexError("page outside store")
        out = np.empty_like(queries)
        with self.lock:
            if self.fd is None: raise RuntimeError("closed store")
            for page,x,y in zip(ids.ravel(),queries.reshape(-1,self.dimension),out.reshape(-1,self.dimension)):
                self.reads += 1
                try:
                    b = os.pread(self.fd,self.block_bytes,int(page)*self.block_bytes)
                    self.returned_bytes += len(b)
                    if len(b) != self.block_bytes: raise OSError("short page read")
                    k,v = decode_block(b)
                    if k.shape[1] != self.dimension: raise ValueError("dimension mismatch")
                    y[:] = residual(k,v,x)
                except Exception:
                    self.failed_reads += 1
                    raise
        return out
