"""Fixed-block parameter stores with explicit probe accounting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class ProbeStats:
    probes: int = 0
    bytes_read: int = 0

    def reset(self) -> None:
        self.probes = 0
        self.bytes_read = 0


class InMemoryParameterStore:
    def __init__(self, blocks: list[bytes]):
        if not blocks:
            raise ValueError("at least one block is required")
        block_bytes = len(blocks[0])
        if block_bytes == 0 or any(len(block) != block_bytes for block in blocks):
            raise ValueError("all blocks must be non-empty and equal sized")
        self.blocks = blocks
        self.block_bytes = block_bytes
        self.num_blocks = len(blocks)
        self.stats = ProbeStats()

    def read_block(self, block_id: int) -> bytes:
        if block_id < 0 or block_id >= self.num_blocks:
            raise IndexError(block_id)
        block = self.blocks[block_id]
        self.stats.probes += 1
        self.stats.bytes_read += len(block)
        return block


class FileParameterStore:
    """Explicit fixed-size pread-backed parameter store.

    This backend intentionally uses os.pread rather than mmap so the experiment
    can account for every logical parameter read. G4 will add an aligned
    O_DIRECT backend for physical cold-storage measurements.
    """

    def __init__(self, path: str | os.PathLike[str], block_bytes: int):
        if block_bytes <= 0:
            raise ValueError("block_bytes must be positive")
        self.path = Path(path)
        self.block_bytes = int(block_bytes)
        size = self.path.stat().st_size
        if size % self.block_bytes:
            raise ValueError("file size must be an integer number of blocks")
        self.num_blocks = size // self.block_bytes
        self._fd = os.open(self.path, os.O_RDONLY)
        self.stats = ProbeStats()

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "FileParameterStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_block(self, block_id: int) -> bytes:
        if self._fd is None:
            raise RuntimeError("store is closed")
        if block_id < 0 or block_id >= self.num_blocks:
            raise IndexError(block_id)
        data = os.pread(self._fd, self.block_bytes, block_id * self.block_bytes)
        if len(data) != self.block_bytes:
            raise IOError("short parameter-block read")
        self.stats.probes += 1
        self.stats.bytes_read += len(data)
        return data
