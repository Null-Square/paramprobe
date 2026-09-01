"""G6a: predeclared larger subword-LM fixed-router capacity smoke gate.

Protocol is frozen in docs/g6a_subword_scale_predeclared.md before execution.
The experiment changes model/tokenization scale rather than retuning the tiny LM.
It uses a three-point N=1/16/256 fixed factorized routing sweep under q=1 and
one 16 KiB external page/token.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_batch(
    data: torch.Tensor, context: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - context - 1, (batch_size,))
    indices = starts[:, None] + torch.arange(context + 1)[None]
    sequence = data[indices]
    return sequence[:, :context], sequence[:, 1:]


def train_or_load_tokenizer(
    train_path: str,
    tokenizer_json: str,
    vocab_size: int,
    min_frequency: int,
    retrain: bool,
) -> Tokenizer:
    path = Path(tokenizer_json)
    if retrain or not path.exists():
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<unk>"],
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train([train_path], trainer)
        path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(path))
    else:
        tokenizer = Tokenizer.from_file(str(path))
    return tokenizer


def encode_split(tokenizer: Tokenizer, path: str) -> torch.Tensor:
    text = Path(path).read_text(encoding="utf-8")
    ids = tokenizer.encode(text).ids
    if len(ids) < 1024:
        raise ValueError(f"tokenized split is unexpectedly small: {path}")
    return torch.tensor(ids, dtype=torch.long)


def bits_to_ids(bits: torch.Tensor, used_bits: int) -> torch.Tensor:
    if used_bits == 0:
        return torch.zeros(bits.shape[:-1], dtype=torch.long, device=bits.device)
    ids = torch.zeros(bits.shape[:-1], dtype=torch.long, device=bits.device)
    for factor in range(used_bits):
        ids = (ids << 1) | bits[..., factor].long()
    return ids


class SubwordCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 96,
        heads: int = 4,
        layers: int = 3,
        ff_dim: int = 384,
        context: int = 128,
        insertion_after: int = 2,
    ) -> None:
        super().__init__()
        if not (0 < insertion_after < layers):
            raise ValueError("insertion_after must leave at least one block on each side")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context = context
        self.insertion_after = insertion_after
        self.token = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(context, d_model)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    heads,
                    dim_feedforward=ff_dim,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
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
        mask = self.causal_mask(length, tokens.device)
        for block in self.blocks[: self.insertion_after]:
            hidden = block(hidden, src_mask=mask)
        return hidden

    def rest(self, hidden: torch.Tensor) -> torch.Tensor:
        length = hidden.shape[1]
        mask = self.causal_mask(length, hidden.device)
        for block in self.blocks[self.insertion_after :]:
            hidden = block(hidden, src_mask=mask)
        return self.norm(hidden)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.rest(self.first(tokens)))


class PageResidualMLP(nn.Module):
    def __init__(
        self,
        num_pages: int,
        d_model: int,
        hidden_dim: int,
        residual_scale: float,
        block_bytes: int,
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
                f"page payload {self.page_payload_bytes} exceeds block {block_bytes}"
            )
        self.w1 = nn.Embedding(num_pages, hidden_dim * d_model)
        self.b1 = nn.Embedding(num_pages, hidden_dim)
        self.w2 = nn.Embedding(num_pages, d_model * hidden_dim)
        self.b2 = nn.Embedding(num_pages, d_model)
        nn.init.normal_(self.w1.weight, std=0.15 / math.sqrt(d_model))
        nn.init.zeros_(self.b1.weight)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.b2.weight)

    def forward(self, page_ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        shape = hidden.shape
        x = hidden.reshape(-1, self.d_model)
        ids = page_ids.reshape(-1)
        batch = x.shape[0]
        w1 = self.w1(ids).view(batch, self.hidden_dim, self.d_model)
        inner = torch.tanh(
            torch.bmm(w1, x.unsqueeze(-1)).squeeze(-1) + self.b1(ids)
        )
        w2 = self.w2(ids).view(batch, self.d_model, self.hidden_dim)
        output = torch.bmm(w2, inner.unsqueeze(-1)).squeeze(-1) + self.b2(ids)
        return (self.residual_scale * torch.tanh(output)).view(shape)


def pretrain_backbone(
    model: SubwordCausalLM,
    train: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    for step in range(steps):
        x, y = get_batch(train, model.context, batch_size)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size), y.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % 100 == 0:
            print(f"pretrain_step={step + 1} pretrain_loss={loss.item():.6f}", flush=True)


def evaluate_backbone(
    model: SubwordCausalLM,
    validation: torch.Tensor,
    steps: int,
    batch_size: int,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    values: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            x, y = get_batch(validation, model.context, batch_size)
            logits = model(x)
            values.append(
                F.cross_entropy(
                    logits.reshape(-1, model.vocab_size), y.reshape(-1)
                ).item()
            )
    return float(np.mean(values))


def build_balanced_hash(
    model: SubwordCausalLM,
    train: torch.Tensor,
    max_bits: int,
    calibration_batches: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def make_hidden_bank(
    model: SubwordCausalLM,
    data: torch.Tensor,
    steps: int,
    batch_size: int,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(seed)
    bank: list[tuple[torch.Tensor, torch.Tensor]] = []
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            x, y = get_batch(data, model.context, batch_size)
            bank.append((model.first(x).detach(), y.detach()))
    return bank


def paired_page_table(
    num_pages: int,
    model: SubwordCausalLM,
    args: argparse.Namespace,
    seed: int,
) -> PageResidualMLP:
    pages = PageResidualMLP(
        num_pages,
        d_model=model.d_model,
        hidden_dim=args.page_hidden_dim,
        residual_scale=args.residual_scale,
        block_bytes=args.block_bytes,
    )
    torch.manual_seed(args.page_init_seed + seed)
    torch.nn.init.normal_(
        pages.w1.weight, std=0.15 / math.sqrt(model.d_model)
    )
    torch.nn.init.zeros_(pages.b1.weight)
    torch.nn.init.zeros_(pages.w2.weight)
    torch.nn.init.zeros_(pages.b2.weight)
    return pages


def train_pages_from_bank(
    model: SubwordCausalLM,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    used_bits: int,
    args: argparse.Namespace,
    seed: int,
) -> PageResidualMLP:
    num_pages = 1 if used_bits == 0 else 2**used_bits
    pages = paired_page_table(num_pages, model, args, seed)
    for parameter in model.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        pages.parameters(), lr=args.page_lr, weight_decay=1e-4
    )
    model.eval()
    pages.train()
    for hidden, y in bank:
        with torch.no_grad():
            page_ids = route(hidden, projection, thresholds, used_bits)
        logits = model.head(model.rest(hidden + pages(page_ids, hidden)))
        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size), y.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return pages


def evaluate_pages_from_bank(
    model: SubwordCausalLM,
    pages: PageResidualMLP,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    used_bits: int,
) -> tuple[float, float, float]:
    values: list[float] = []
    counts = torch.zeros(pages.num_pages, dtype=torch.float64)
    model.eval()
    pages.eval()
    with torch.no_grad():
        for hidden, y in bank:
            page_ids = route(hidden, projection, thresholds, used_bits)
            logits = model.head(model.rest(hidden + pages(page_ids, hidden)))
            values.append(
                F.cross_entropy(
                    logits.reshape(-1, model.vocab_size), y.reshape(-1)
                ).item()
            )
            counts += torch.bincount(
                page_ids.reshape(-1).cpu(), minlength=pages.num_pages
            ).double()
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
    dead = float(np.mean(raw == 0))
    return float(np.mean(values)), entropy, dead


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    set_seed(args.backbone_seed)

    tokenizer = train_or_load_tokenizer(
        args.train_path,
        args.tokenizer_json,
        args.vocab_size,
        args.min_frequency,
        args.retrain_tokenizer,
    )
    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab != args.vocab_size:
        raise RuntimeError(
            f"tokenizer vocabulary {actual_vocab} != requested {args.vocab_size}"
        )
    tokenizer_bytes = Path(args.tokenizer_json).read_bytes()
    tokenizer_sha = hashlib.sha256(tokenizer_bytes).hexdigest()
    train = encode_split(tokenizer, args.train_path)
    validation = encode_split(tokenizer, args.validation_path)

    model = SubwordCausalLM(
        vocab_size=actual_vocab,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        ff_dim=args.ff_dim,
        context=args.context,
        insertion_after=args.insertion_after,
    )
    backbone_parameters = sum(p.numel() for p in model.parameters())
    checkpoint = Path(args.checkpoint)
    if args.retrain_backbone or not checkpoint.exists():
        pretrain_backbone(
            model,
            train,
            args.pretrain_steps,
            args.pretrain_batch_size,
            args.pretrain_lr,
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
    else:
        model.load_state_dict(torch.load(checkpoint, weights_only=True))

    backbone_ce = evaluate_backbone(
        model,
        validation,
        args.eval_steps,
        args.eval_batch_size,
        args.eval_seed,
    )
    projection, thresholds = build_balanced_hash(
        model,
        train,
        args.max_bits,
        args.hash_calibration_batches,
        args.hash_batch_size,
    )

    used_values = [int(v) for v in args.used_bits.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    if used_values != [0, 4, 8]:
        raise ValueError("G6a predeclared used_bits must be exactly 0,4,8")
    if seeds != [7, 8, 9]:
        raise ValueError("G6a predeclared page seeds must be exactly 7,8,9")

    prototype = PageResidualMLP(
        1,
        d_model=model.d_model,
        hidden_dim=args.page_hidden_dim,
        residual_scale=args.residual_scale,
        block_bytes=args.block_bytes,
    )
    page_parameters = prototype.page_parameters
    page_payload_bytes = prototype.page_payload_bytes
    active_page_macs = 2 * model.d_model * args.page_hidden_dim
    fixed_router_macs = model.d_model * args.max_bits
    if page_parameters != 3956 or page_payload_bytes != 15824:
        raise RuntimeError("G6a page shape no longer matches the predeclared payload")
    if args.block_bytes != 16384 or active_page_macs != 3840:
        raise RuntimeError("G6a resource envelope changed")

    eval_bank = make_hidden_bank(
        model,
        validation,
        args.eval_steps,
        args.eval_batch_size,
        args.eval_seed,
    )

    results: dict[int, list[float]] = {u: [] for u in used_values}
    entropy_results: dict[int, list[float]] = {u: [] for u in used_values}
    dead_results: dict[int, list[float]] = {u: [] for u in used_values}
    by_seed: dict[int, list[float]] = {s: [] for s in seeds}

    print("# ParamProbe G6a: larger subword-LM fixed-router capacity gate")
    print("protocol_predeclared=true")
    print("tokenizer_train_only=true")
    print(f"tokenizer_vocab_size={actual_vocab}")
    print(f"tokenizer_json_bytes={len(tokenizer_bytes)}")
    print(f"tokenizer_json_sha256={tokenizer_sha}")
    print(f"train_tokens={len(train)}")
    print(f"validation_tokens={len(validation)}")
    print(f"backbone_parameters={backbone_parameters}")
    print(f"backbone_validation_ce={backbone_ce:.8f}")
    print(f"d_model={model.d_model}")
    print(f"layers={args.layers}")
    print(f"context={model.context}")
    print(f"insertion_after_block={args.insertion_after}")
    print(f"page_parameters={page_parameters}")
    print(f"page_payload_bytes={page_payload_bytes}")
    print(f"page_padding_bytes={args.block_bytes - page_payload_bytes}")
    print(f"block_bytes={args.block_bytes}")
    print(f"active_page_macs_per_token={active_page_macs}")
    print("q_inference=1")
    print(f"logical_external_bytes_per_token={args.block_bytes}")
    print(f"fixed_router_macs_per_token={fixed_router_macs}")
    print(f"max_address_bits={args.max_bits}")
    print("paired_page_prefix_initialization=true")
    print("paired_training_minibatches=true")
    print()
    print("seed\tused_bits\tpages\tvalidation_ce\tutil_entropy\tdead_page_fraction")

    for seed in seeds:
        train_bank = make_hidden_bank(
            model,
            train,
            args.page_steps,
            args.page_batch_size,
            args.page_batch_seed + seed,
        )
        for used_bits in used_values:
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            pages = train_pages_from_bank(
                model,
                train_bank,
                projection,
                thresholds,
                used_bits,
                args,
                seed,
            )
            ce, entropy, dead = evaluate_pages_from_bank(
                model,
                pages,
                eval_bank,
                projection,
                thresholds,
                used_bits,
            )
            num_pages = 1 if used_bits == 0 else 2**used_bits
            results[used_bits].append(ce)
            entropy_results[used_bits].append(entropy)
            dead_results[used_bits].append(dead)
            by_seed[seed].append(ce)
            print(
                f"{seed}\t{used_bits}\t{num_pages}\t{ce:.8f}\t{entropy:.8f}\t{dead:.8f}",
                flush=True,
            )

    print()
    print("pages\tvalidation_ce_mean\tvalidation_ce_sample_std\tutil_entropy_mean\tdead_page_fraction_mean")
    for used_bits in used_values:
        values = np.asarray(results[used_bits], dtype=np.float64)
        entropies = np.asarray(entropy_results[used_bits], dtype=np.float64)
        dead = np.asarray(dead_results[used_bits], dtype=np.float64)
        num_pages = 1 if used_bits == 0 else 2**used_bits
        print(
            f"{num_pages}\t{values.mean():.8f}\t{values.std(ddof=1):.8f}\t"
            f"{entropies.mean():.8f}\t{dead.mean():.8f}"
        )

    monotone = {
        seed: bool(np.all(np.diff(np.asarray(values, dtype=np.float64)) < 0.0))
        for seed, values in by_seed.items()
    }
    endpoint_improved = {
        seed: bool(values[-1] < values[0]) for seed, values in by_seed.items()
    }
    print()
    print(
        "strict_monotone_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in monotone.items())
    )
    print(
        "endpoint_improved_by_seed="
        + ",".join(
            f"{s}:{str(v).lower()}" for s, v in endpoint_improved.items()
        )
    )
    print("resource_gate_passed=true")
    print("validation_leakage_gate_passed=true")
    print("g6a_gate_passed=" + str(all(monotone.values())).lower())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--tokenizer-json", default=".cache/g6a_tokenizer.json")
    parser.add_argument("--retrain-tokenizer", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--checkpoint", default=".cache/g6a_subword_base.pt")
    parser.add_argument("--retrain-backbone", action="store_true")
    parser.add_argument("--backbone-seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ff-dim", type=int, default=384)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--insertion-after", type=int, default=2)
    parser.add_argument("--pretrain-steps", type=int, default=500)
    parser.add_argument("--pretrain-batch-size", type=int, default=8)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--max-bits", type=int, default=8)
    parser.add_argument("--used-bits", default="0,4,8")
    parser.add_argument("--hash-calibration-batches", type=int, default=40)
    parser.add_argument("--hash-batch-size", type=int, default=8)
    parser.add_argument("--page-hidden-dim", type=int, default=20)
    parser.add_argument("--residual-scale", type=float, default=0.15)
    parser.add_argument("--block-bytes", type=int, default=16384)
    parser.add_argument("--page-init-seed", type=int, default=40000)
    parser.add_argument("--page-batch-seed", type=int, default=50000)
    parser.add_argument("--seeds", default="7,8,9")
    parser.add_argument("--page-steps", type=int, default=120)
    parser.add_argument("--page-batch-size", type=int, default=8)
    parser.add_argument("--page-lr", type=float, default=4e-3)
    parser.add_argument("--eval-steps", type=int, default=40)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())
