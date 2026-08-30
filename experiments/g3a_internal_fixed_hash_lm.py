"""G3a: tiny language-model precheck with an internal one-page ParamProbe layer.

This experiment is intentionally conservative. It freezes a two-block byte-level
Transformer and inserts ParamProbe between block 1 and block 2. Routing is a
fixed balanced context hash rather than a learned task router so that the first
language-model capacity test isolates the external operator capacity/insertion
question from routing optimization.

For every capacity point:

- the pretrained Transformer is identical and frozen;
- the full six-factor hash metadata and routing MACs are identical;
- the page MLP is identical (48 -> 10 -> 48);
- one page is selected per token at inference (`q=1`);
- each page contains 1,018 FP32 parameters = 4,072 payload bytes and is padded
  to one 4,096-byte external parameter block;
- only the number of inactive external pages changes.

The default no-argument corpus fallback concatenates local Python/PyTorch source
files and is only for an environment-local diagnostic. For a reproducible
benchmark, provide `--text-path` (for example a local Tiny Shakespeare file).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_corpus(text_path: str | None, limit: int) -> bytes:
    if text_path is not None:
        data = Path(text_path).read_bytes()
        if len(data) < 1024:
            raise ValueError("text corpus is unexpectedly small")
        return data[:limit]

    roots = [os.path.dirname(os.__file__), os.path.dirname(torch.__file__)]
    chunks: list[bytes] = []
    total = 0
    for root in roots:
        for path in sorted(glob.glob(root + "/**/*.py", recursive=True)):
            try:
                payload = Path(path).read_bytes()
            except OSError:
                continue
            if not payload:
                continue
            chunks.append(payload + b"\n")
            total += len(payload) + 1
            if total >= limit:
                return b"".join(chunks)[:limit]
    data = b"".join(chunks)
    if len(data) < 1024:
        raise RuntimeError("could not build fallback corpus")
    return data[:limit]


def get_batch(data: torch.Tensor, context: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - context - 1, (batch_size,))
    indices = starts[:, None] + torch.arange(context + 1)[None]
    sequence = data[indices]
    return sequence[:, :context], sequence[:, 1:]


def bits_to_ids(bits: torch.Tensor, used_bits: int) -> torch.Tensor:
    if used_bits == 0:
        return torch.zeros(bits.shape[:-1], dtype=torch.long, device=bits.device)
    ids = torch.zeros(bits.shape[:-1], dtype=torch.long, device=bits.device)
    for factor in range(used_bits):
        ids = (ids << 1) | bits[..., factor].long()
    return ids


class TwoBlockByteLM(nn.Module):
    def __init__(self, d_model: int = 48, heads: int = 4, context: int = 64) -> None:
        super().__init__()
        self.d_model = d_model
        self.context = context
        self.token = nn.Embedding(256, d_model)
        self.position = nn.Embedding(context, d_model)
        self.block1 = nn.TransformerEncoderLayer(
            d_model,
            heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.block2 = nn.TransformerEncoderLayer(
            d_model,
            heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256, bias=False)
        self.head.weight = self.token.weight

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def first(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token(tokens) + self.position(positions)[None]
        return self.block1(
            hidden, src_mask=self.causal_mask(length, tokens.device)
        )

    def rest(self, hidden: torch.Tensor) -> torch.Tensor:
        length = hidden.shape[1]
        hidden = self.block2(
            hidden, src_mask=self.causal_mask(length, hidden.device)
        )
        return self.norm(hidden)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.rest(self.first(tokens)))


class PageResidualMLP(nn.Module):
    def __init__(
        self,
        num_pages: int,
        d_model: int = 48,
        hidden_dim: int = 10,
        residual_scale: float = 0.15,
        block_bytes: int = 4096,
    ) -> None:
        super().__init__()
        self.num_pages = num_pages
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.residual_scale = residual_scale
        self.page_parameters = (
            hidden_dim * d_model
            + hidden_dim
            + d_model * hidden_dim
            + d_model
        )
        self.page_payload_bytes = 4 * self.page_parameters
        if self.page_payload_bytes > block_bytes:
            raise ValueError(
                f"page MLP payload {self.page_payload_bytes} exceeds {block_bytes} bytes"
            )

        self.w1 = nn.Embedding(num_pages, hidden_dim * d_model)
        self.b1 = nn.Embedding(num_pages, hidden_dim)
        self.w2 = nn.Embedding(num_pages, d_model * hidden_dim)
        self.b2 = nn.Embedding(num_pages, d_model)
        nn.init.normal_(self.w1.weight, std=0.15 / math.sqrt(d_model))
        nn.init.zeros_(self.b1.weight)
        # Exact no-op initialization protects the frozen backbone initially.
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.b2.weight)

    def forward(self, page_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        original_shape = hidden.shape
        x = hidden.reshape(-1, self.d_model)
        flat_ids = page_ids.reshape(-1)
        batch = x.shape[0]

        w1 = self.w1(flat_ids).view(batch, self.hidden_dim, self.d_model)
        inner = torch.tanh(
            torch.bmm(w1, x.unsqueeze(-1)).squeeze(-1) + self.b1(flat_ids)
        )
        w2 = self.w2(flat_ids).view(batch, self.d_model, self.hidden_dim)
        output = (
            torch.bmm(w2, inner.unsqueeze(-1)).squeeze(-1) + self.b2(flat_ids)
        )
        return (self.residual_scale * torch.tanh(output)).view(original_shape)


def pretrain_backbone(
    model: TwoBlockByteLM,
    train: torch.Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    model.train()
    for _ in range(steps):
        x, y = get_batch(train, model.context, batch_size)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def evaluate_backbone(
    model: TwoBlockByteLM,
    validation: torch.Tensor,
    steps: int,
    batch_size: int,
) -> float:
    torch.manual_seed(1234)
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            x, y = get_batch(validation, model.context, batch_size)
            values.append(
                F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1)).item()
            )
    return float(np.mean(values))


def build_balanced_hash(
    model: TwoBlockByteLM,
    train: torch.Tensor,
    max_bits: int,
    calibration_batches: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Allocate the maximum hash width once. All capacity conditions reuse it and
    # take prefixes, so resident metadata and routing matrix MACs are fixed.
    torch.manual_seed(999)
    projection, _ = torch.linalg.qr(
        torch.randn(model.d_model, max_bits), mode="reduced"
    )
    projected: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for _ in range(calibration_batches):
            x, _ = get_batch(train, model.context, batch_size)
            hidden = model.first(x)
            projected.append((hidden @ projection).reshape(-1, max_bits))
    values = torch.cat(projected, dim=0)
    thresholds = values.median(dim=0).values
    return projection, thresholds


def route(
    hidden: torch.Tensor,
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    used_bits: int,
) -> torch.Tensor:
    if used_bits == 0:
        return torch.zeros(hidden.shape[:-1], dtype=torch.long, device=hidden.device)
    bits = (hidden @ projection)[..., :used_bits] > thresholds[:used_bits]
    return bits_to_ids(bits, used_bits)


def train_pages(
    model: TwoBlockByteLM,
    train: torch.Tensor,
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    used_bits: int,
    block_bytes: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> PageResidualMLP:
    num_pages = 1 if used_bits == 0 else 2**used_bits
    pages = PageResidualMLP(
        num_pages,
        d_model=model.d_model,
        hidden_dim=10,
        residual_scale=0.15,
        block_bytes=block_bytes,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        pages.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    model.eval()
    pages.train()

    for _ in range(steps):
        x, y = get_batch(train, model.context, batch_size)
        with torch.no_grad():
            hidden = model.first(x)
            page_ids = route(hidden, projection, thresholds, used_bits)
        adapted = hidden + pages(page_ids, hidden)
        logits = model.head(model.rest(adapted))
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return pages


def evaluate_pages(
    model: TwoBlockByteLM,
    pages: PageResidualMLP,
    validation: torch.Tensor,
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    used_bits: int,
    steps: int,
    batch_size: int,
) -> tuple[float, float, float]:
    torch.manual_seed(1234)
    values: list[float] = []
    counts = torch.zeros(pages.num_pages)
    model.eval()
    pages.eval()
    with torch.no_grad():
        for _ in range(steps):
            x, y = get_batch(validation, model.context, batch_size)
            hidden = model.first(x)
            page_ids = route(hidden, projection, thresholds, used_bits)
            logits = model.head(model.rest(hidden + pages(page_ids, hidden)))
            values.append(
                F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).item()
            )
            counts += torch.bincount(
                page_ids.reshape(-1).cpu(), minlength=pages.num_pages
            )

    raw = counts.numpy()
    positive = raw[raw > 0]
    if pages.num_pages == 1:
        entropy = 1.0
    else:
        probabilities = positive / positive.sum()
        entropy = float(
            -(probabilities * np.log(probabilities)).sum()
            / np.log(pages.num_pages)
        )
    dead_fraction = float(np.mean(raw == 0))
    return float(np.mean(values)), entropy, dead_fraction


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    set_seed(args.backbone_seed)
    raw = load_corpus(args.text_path, args.corpus_bytes)
    data = torch.tensor(list(raw), dtype=torch.long)
    split = int(0.9 * len(data))
    train, validation = data[:split], data[split:]

    model = TwoBlockByteLM(
        d_model=args.d_model, heads=args.heads, context=args.context
    )
    checkpoint = Path(args.checkpoint)
    if args.retrain_backbone or not checkpoint.exists():
        pretrain_backbone(
            model,
            train,
            steps=args.pretrain_steps,
            batch_size=args.pretrain_batch_size,
            learning_rate=args.pretrain_lr,
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
    else:
        model.load_state_dict(torch.load(checkpoint, weights_only=True))

    base_ce = evaluate_backbone(
        model, validation, args.eval_steps, args.eval_batch_size
    )
    projection, thresholds = build_balanced_hash(
        model,
        train,
        max_bits=args.max_bits,
        calibration_batches=args.hash_calibration_batches,
        batch_size=args.hash_batch_size,
    )

    router_scalars = model.d_model * args.max_bits + args.max_bits
    routing_macs = model.d_model * args.max_bits
    page_parameters = 10 * model.d_model + 10 + model.d_model * 10 + model.d_model
    page_payload_bytes = 4 * page_parameters
    active_page_macs = 2 * model.d_model * 10

    print("# ParamProbe G3a: internal fixed-hash language-model precheck")
    print(f"corpus_bytes={len(raw)}")
    print(f"backbone_validation_ce={base_ce:.8f}")
    print(f"max_bits={args.max_bits}")
    print(f"fixed_router_scalars={router_scalars}")
    print(f"fixed_routing_macs_per_token={routing_macs}")
    print(f"page_parameters={page_parameters}")
    print(f"page_payload_bytes={page_payload_bytes}")
    print(f"block_bytes={args.block_bytes}")
    print(f"active_page_macs_per_token={active_page_macs}")
    print("q_inference=1")
    print(f"logical_external_bytes_per_token={args.block_bytes}")
    print()
    print("seed\tused_bits\tpages\tvalidation_ce\tutil_entropy\tdead_pages")

    seeds = [int(value) for value in args.seeds.split(",")]
    used_bits_values = [int(value) for value in args.used_bits.split(",")]
    for seed in seeds:
        for used_bits in used_bits_values:
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            set_seed(seed)
            pages = train_pages(
                model,
                train,
                projection,
                thresholds,
                used_bits,
                block_bytes=args.block_bytes,
                steps=args.page_steps,
                batch_size=args.page_batch_size,
                learning_rate=args.page_lr,
            )
            ce, entropy, dead = evaluate_pages(
                model,
                pages,
                validation,
                projection,
                thresholds,
                used_bits,
                steps=args.eval_steps,
                batch_size=args.eval_batch_size,
            )
            num_pages = 1 if used_bits == 0 else 2**used_bits
            print(
                f"{seed}\t{used_bits}\t{num_pages}\t{ce:.8f}\t"
                f"{entropy:.8f}\t{dead:.8f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-path", type=str, default=None)
    parser.add_argument("--corpus-bytes", type=int, default=1_200_000)
    parser.add_argument("--checkpoint", type=str, default=".cache/paramprobe_g3a_base.pt")
    parser.add_argument("--retrain-backbone", action="store_true")
    parser.add_argument("--backbone-seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--pretrain-batch-size", type=int, default=32)
    parser.add_argument("--pretrain-lr", type=float, default=3e-3)
    parser.add_argument("--max-bits", type=int, default=6)
    parser.add_argument("--hash-calibration-batches", type=int, default=50)
    parser.add_argument("--hash-batch-size", type=int, default=24)
    parser.add_argument("--used-bits", type=str, default="0,2,4,6")
    parser.add_argument("--seeds", type=str, default="7,8,9")
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--page-steps", type=int, default=180)
    parser.add_argument("--page-batch-size", type=int, default=24)
    parser.add_argument("--page-lr", type=float, default=4e-3)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())
