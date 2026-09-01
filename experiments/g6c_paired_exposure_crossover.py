"""G6c: independent paired training-exposure crossover confirmation.

Predeclared in docs/g6c_paired_exposure_crossover_predeclared.md before this
file was created.  The exact G6a tokenizer/backbone are frozen.  Fresh page
seeds 10/11/12 compare N=16@120 against N=256@120..1920 inside one run.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import g6a_subword_scale_lm as g6a


TOKENIZER_SHA256 = "230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb"
BACKBONE_SHA256 = "a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049"
CHECKPOINTS = (120, 240, 480, 960, 1920)
SEEDS = (10, 11, 12)


def sha256_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_assets(args: argparse.Namespace) -> tuple[Tokenizer, g6a.SubwordCausalLM]:
    tok_sha = sha256_path(args.tokenizer_json)
    model_sha = sha256_path(args.checkpoint)
    if tok_sha != TOKENIZER_SHA256:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    if model_sha != BACKBONE_SHA256:
        raise RuntimeError(f"backbone SHA mismatch: {model_sha}")
    tokenizer = Tokenizer.from_file(args.tokenizer_json)
    if tokenizer.get_vocab_size() != 1024:
        raise RuntimeError("frozen tokenizer vocabulary changed")
    model = g6a.SubwordCausalLM(
        vocab_size=1024,
        d_model=96,
        heads=4,
        layers=3,
        ff_dim=384,
        context=128,
        insertion_after=2,
    )
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return tokenizer, model


def init_pages(num_pages: int, seed: int) -> g6a.PageResidualMLP:
    pages = g6a.PageResidualMLP(
        num_pages,
        d_model=96,
        hidden_dim=20,
        residual_scale=0.15,
        block_bytes=16384,
    )
    torch.manual_seed(40000 + seed)
    torch.nn.init.normal_(pages.w1.weight, std=0.15 / math.sqrt(96))
    torch.nn.init.zeros_(pages.b1.weight)
    torch.nn.init.zeros_(pages.w2.weight)
    torch.nn.init.zeros_(pages.b2.weight)
    return pages


def eval_pages(
    model: g6a.SubwordCausalLM,
    pages: g6a.PageResidualMLP,
    eval_bank: list[tuple[torch.Tensor, torch.Tensor]],
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    used_bits: int,
) -> float:
    vals: list[float] = []
    pages.eval()
    with torch.no_grad():
        for hidden, y in eval_bank:
            ids = g6a.route(hidden, projection, thresholds, used_bits)
            logits = model.head(model.rest(hidden + pages(ids, hidden)))
            vals.append(
                F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1)).item()
            )
    return float(np.mean(vals))


def route_stats(
    eval_bank: list[tuple[torch.Tensor, torch.Tensor]],
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    used_bits: int,
) -> tuple[float, float]:
    n = 2**used_bits
    counts = torch.zeros(n, dtype=torch.float64)
    with torch.no_grad():
        for hidden, _ in eval_bank:
            ids = g6a.route(hidden, projection, thresholds, used_bits)
            counts += torch.bincount(ids.reshape(-1).cpu(), minlength=n).double()
    raw = counts.numpy()
    positive = raw[raw > 0]
    probs = positive / positive.sum()
    entropy = float(-(probs * np.log(probs)).sum() / np.log(n))
    return entropy, float(np.mean(raw == 0))


def train_n16(
    model: g6a.SubwordCausalLM,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
    eval_bank: list[tuple[torch.Tensor, torch.Tensor]],
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    seed: int,
) -> float:
    pages = init_pages(16, seed)
    optimizer = torch.optim.AdamW(pages.parameters(), lr=4e-3, weight_decay=1e-4)
    pages.train()
    for hidden, y in bank[:120]:
        ids = g6a.route(hidden, projection, thresholds, 4)
        logits = model.head(model.rest(hidden + pages(ids, hidden)))
        loss = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return eval_pages(model, pages, eval_bank, projection, thresholds, 4)


def train_n256(
    model: g6a.SubwordCausalLM,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
    eval_bank: list[tuple[torch.Tensor, torch.Tensor]],
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    seed: int,
) -> dict[int, float]:
    pages = init_pages(256, seed)
    optimizer = torch.optim.AdamW(pages.parameters(), lr=4e-3, weight_decay=1e-4)
    out: dict[int, float] = {}
    pages.train()
    for step, (hidden, y) in enumerate(bank, start=1):
        ids = g6a.route(hidden, projection, thresholds, 8)
        logits = model.head(model.rest(hidden + pages(ids, hidden)))
        loss = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in CHECKPOINTS:
            out[step] = eval_pages(
                model, pages, eval_bank, projection, thresholds, 8
            )
            pages.train()
            print(
                f"n256_checkpoint\t{seed}\t{step}\t"
                f"{step * 8 * 128 / 256:.1f}\t{out[step]:.8f}",
                flush=True,
            )
    return out


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    tokenizer, model = load_assets(args)
    train = g6a.encode_split(tokenizer, args.train_path)
    valid = g6a.encode_split(tokenizer, args.validation_path)
    if len(train) != 4_254_523 or len(valid) != 445_470:
        raise RuntimeError("tokenized split sizes changed")

    projection, thresholds = g6a.build_balanced_hash(
        model, train, max_bits=8, calibration_batches=40, batch_size=8
    )
    eval_bank = g6a.make_hidden_bank(model, valid, 40, 8, 1234)
    entropy16, dead16 = route_stats(eval_bank, projection, thresholds, 4)
    entropy256, dead256 = route_stats(eval_bank, projection, thresholds, 8)

    page_parameters = 20 * 96 + 20 + 96 * 20 + 96
    if page_parameters != 3956 or 4 * page_parameters != 15824:
        raise RuntimeError("page resource envelope changed")
    if 2 * 96 * 20 != 3840 or 96 * 8 != 768:
        raise RuntimeError("active/router MAC envelope changed")

    print("# ParamProbe G6c: paired exposure crossover")
    print("protocol_predeclared=true")
    print("fresh_page_seeds=10,11,12")
    print(f"tokenizer_sha256={sha256_path(args.tokenizer_json)}")
    print(f"backbone_sha256={sha256_path(args.checkpoint)}")
    print("q_inference=1")
    print("block_bytes=16384")
    print("logical_external_bytes_per_token=16384")
    print("page_parameters=3956")
    print("page_payload_bytes=15824")
    print("active_page_macs_per_token=3840")
    print("fixed_router_macs_per_token=768")
    print(f"n16_util_entropy={entropy16:.8f}")
    print(f"n16_dead_page_fraction={dead16:.8f}")
    print(f"n256_util_entropy={entropy256:.8f}")
    print(f"n256_dead_page_fraction={dead256:.8f}")
    print()

    n16: dict[int, float] = {}
    n256: dict[int, dict[int, float]] = {}
    for seed in SEEDS:
        # Exact G6a hidden-bank implementation.  The same first 120 frozen
        # hidden/target minibatches feed both capacities for this seed.
        bank = g6a.make_hidden_bank(model, train, 1920, 8, 50000 + seed)
        n16[seed] = train_n16(
            model, bank, eval_bank, projection, thresholds, seed
        )
        print(f"n16_reference\t{seed}\t120\t7680.0\t{n16[seed]:.8f}", flush=True)
        n256[seed] = train_n256(
            model, bank, eval_bank, projection, thresholds, seed
        )
        del bank

    low_exposure_direction: dict[int, bool] = {}
    crossover: dict[int, bool] = {}
    within_improve: dict[int, bool] = {}
    checkpoint_monotone: dict[int, bool] = {}

    print()
    print("seed\tn16_120\tn256_120\tn256_1920\tlow_exposure_direction\tcrossover")
    for seed in SEEDS:
        n16_ce = n16[seed]
        early = n256[seed][120]
        final = n256[seed][1920]
        low_exposure_direction[seed] = n16_ce < early
        crossover[seed] = final < n16_ce
        within_improve[seed] = final < early
        seq = np.asarray([n256[seed][s] for s in CHECKPOINTS])
        checkpoint_monotone[seed] = bool(np.all(np.diff(seq) < 0.0))
        print(
            f"{seed}\t{n16_ce:.8f}\t{early:.8f}\t{final:.8f}\t"
            f"{str(low_exposure_direction[seed]).lower()}\t"
            f"{str(crossover[seed]).lower()}"
        )

    mean16 = float(np.mean([n16[s] for s in SEEDS]))
    print()
    print(f"n16_120_mean={mean16:.8f}")
    print("n256_steps\tmean_assignments_per_page\tce_mean\tce_sample_std")
    for step in CHECKPOINTS:
        vals = np.asarray([n256[s][step] for s in SEEDS], dtype=np.float64)
        print(
            f"{step}\t{step * 8 * 128 / 256:.1f}\t"
            f"{vals.mean():.8f}\t{vals.std(ddof=1):.8f}"
        )

    print()
    print(
        "low_exposure_direction_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in low_exposure_direction.items())
    )
    print(
        "crossover_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in crossover.items())
    )
    print(
        "within_n256_improvement_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in within_improve.items())
    )
    print(
        "n256_checkpoint_monotone_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in checkpoint_monotone.items())
    )
    print("resource_gate_passed=true")
    confirmed = (
        all(low_exposure_direction.values())
        and all(crossover.values())
        and all(within_improve.values())
    )
    print(f"g6c_exposure_crossover_confirmed={str(confirmed).lower()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", required=True)
    p.add_argument("--validation-path", required=True)
    p.add_argument("--tokenizer-json", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threads", type=int, default=8)
    run(p.parse_args())
