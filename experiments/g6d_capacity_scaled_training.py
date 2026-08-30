"""G6d: larger subword-LM capacity sweep with explicit training-budget scaling.

Predeclared in docs/g6d_capacity_scaled_training_predeclared.md before this file
was created.  The exact archived G6a tokenizer/backbone and fixed factorized
router are reused.  Inference resources remain fixed across N; page-training
steps follow the declared exposure-floor rule.
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
SEEDS = (13, 14, 15)
USED_BITS = (0, 2, 4, 6, 8)
BATCH_TOKENS = 8 * 128
EXPOSURE_FLOOR = 7680
MIN_STEPS = 120


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


def num_pages(used_bits: int) -> int:
    return 1 if used_bits == 0 else 2**used_bits


def budget_steps(used_bits: int) -> int:
    n = num_pages(used_bits)
    exposure_steps = math.ceil(EXPOSURE_FLOOR * n / BATCH_TOKENS)
    return max(MIN_STEPS, exposure_steps)


def init_pages(n: int, seed: int) -> g6a.PageResidualMLP:
    pages = g6a.PageResidualMLP(
        n,
        d_model=96,
        hidden_dim=20,
        residual_scale=0.15,
        block_bytes=16384,
    )
    # Explicit post-construction initialization makes prefix rows comparable
    # across differently sized page tables.
    torch.manual_seed(40000 + seed)
    torch.nn.init.normal_(pages.w1.weight, std=0.15 / math.sqrt(96))
    torch.nn.init.zeros_(pages.b1.weight)
    torch.nn.init.zeros_(pages.w2.weight)
    torch.nn.init.zeros_(pages.b2.weight)
    return pages


def route_stats(
    eval_bank: list[tuple[torch.Tensor, torch.Tensor]],
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    used_bits: int,
) -> tuple[float, float]:
    n = num_pages(used_bits)
    counts = torch.zeros(n, dtype=torch.float64)
    with torch.no_grad():
        for hidden, _ in eval_bank:
            ids = g6a.route(hidden, projection, thresholds, used_bits)
            counts += torch.bincount(ids.reshape(-1).cpu(), minlength=n).double()
    raw = counts.numpy()
    if n == 1:
        entropy = 1.0
    else:
        positive = raw[raw > 0]
        probs = positive / positive.sum()
        entropy = float(-(probs * np.log(probs)).sum() / np.log(n))
    dead = float(np.mean(raw == 0))
    return entropy, dead


def evaluate(
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


def train_capacity(
    model: g6a.SubwordCausalLM,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
    eval_bank: list[tuple[torch.Tensor, torch.Tensor]],
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    used_bits: int,
    seed: int,
) -> float:
    n = num_pages(used_bits)
    steps = budget_steps(used_bits)
    pages = init_pages(n, seed)
    optimizer = torch.optim.AdamW(pages.parameters(), lr=4e-3, weight_decay=1e-4)
    pages.train()
    for hidden, y in bank[:steps]:
        ids = g6a.route(hidden, projection, thresholds, used_bits)
        logits = model.head(model.rest(hidden + pages(ids, hidden)))
        loss = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return evaluate(model, pages, eval_bank, projection, thresholds, used_bits)


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

    expected_steps = {0: 120, 2: 120, 4: 120, 6: 480, 8: 1920}
    actual_steps = {u: budget_steps(u) for u in USED_BITS}
    if actual_steps != expected_steps:
        raise RuntimeError(f"training-budget rule changed: {actual_steps}")

    page_parameters = 20 * 96 + 20 + 96 * 20 + 96
    page_payload = 4 * page_parameters
    active_page_macs = 2 * 96 * 20
    fixed_router_macs = 96 * 8
    if (page_parameters, page_payload, active_page_macs, fixed_router_macs) != (
        3956,
        15824,
        3840,
        768,
    ):
        raise RuntimeError("frozen inference resource envelope changed")

    stats = {
        u: route_stats(eval_bank, projection, thresholds, u) for u in USED_BITS
    }

    print("# ParamProbe G6d: capacity-scaled page-training sweep")
    print("protocol_predeclared=true")
    print("fresh_page_seeds=13,14,15")
    print(f"tokenizer_sha256={sha256_path(args.tokenizer_json)}")
    print(f"backbone_sha256={sha256_path(args.checkpoint)}")
    print(f"train_tokens={len(train)}")
    print(f"validation_tokens={len(valid)}")
    print("q_inference=1")
    print("block_bytes=16384")
    print("logical_external_bytes_per_token=16384")
    print("page_parameters=3956")
    print("page_payload_bytes=15824")
    print("active_page_macs_per_token=3840")
    print("fixed_router_macs_per_token=768")
    print("max_address_bits=8")
    print("training_exposure_floor_per_page=7680")
    print("training_budget_rule=steps(N)=max(120,ceil(7680*N/1024))")
    print()
    print("pages\tused_bits\tsteps\ttotal_assignments\tmean_assignments_per_page\tutil_entropy\tdead_page_fraction")
    for u in USED_BITS:
        n = num_pages(u)
        steps = actual_steps[u]
        total = steps * BATCH_TOKENS
        mean_per_page = total / n
        entropy, dead = stats[u]
        print(
            f"{n}\t{u}\t{steps}\t{total}\t{mean_per_page:.1f}\t"
            f"{entropy:.8f}\t{dead:.8f}"
        )

    results: dict[int, list[float]] = {u: [] for u in USED_BITS}
    by_seed: dict[int, list[float]] = {s: [] for s in SEEDS}
    print()
    print("seed\tused_bits\tpages\tsteps\tvalidation_ce")
    max_steps = max(actual_steps.values())
    for seed in SEEDS:
        # One frozen hidden-state bank per seed.  Every capacity uses a prefix,
        # so the first 120 minibatches are exactly paired across the sweep.
        bank = g6a.make_hidden_bank(model, train, max_steps, 8, 50000 + seed)
        for u in USED_BITS:
            ce = train_capacity(
                model,
                bank,
                eval_bank,
                projection,
                thresholds,
                u,
                seed,
            )
            results[u].append(ce)
            by_seed[seed].append(ce)
            print(
                f"{seed}\t{u}\t{num_pages(u)}\t{actual_steps[u]}\t{ce:.8f}",
                flush=True,
            )
        del bank

    means: list[float] = []
    print()
    print("pages\tsteps\tvalidation_ce_mean\tvalidation_ce_sample_std")
    for u in USED_BITS:
        vals = np.asarray(results[u], dtype=np.float64)
        means.append(float(vals.mean()))
        print(
            f"{num_pages(u)}\t{actual_steps[u]}\t{vals.mean():.8f}\t"
            f"{vals.std(ddof=1):.8f}"
        )

    mean_monotone = bool(np.all(np.diff(np.asarray(means)) < 0.0))
    endpoint_beats_one = {
        seed: by_seed[seed][-1] < by_seed[seed][0] for seed in SEEDS
    }
    endpoint_beats_16 = {
        seed: by_seed[seed][-1] < by_seed[seed][2] for seed in SEEDS
    }
    full_monotone = {
        seed: bool(np.all(np.diff(np.asarray(by_seed[seed])) < 0.0))
        for seed in SEEDS
    }

    print()
    print(f"mean_curve_strictly_monotone={str(mean_monotone).lower()}")
    print(
        "n256_beats_n1_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in endpoint_beats_one.items())
    )
    print(
        "n256_beats_n16_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in endpoint_beats_16.items())
    )
    print(
        "full_monotone_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in full_monotone.items())
    )
    print("resource_gate_passed=true")
    print("training_budget_gate_passed=true")
    passed = (
        mean_monotone
        and all(endpoint_beats_one.values())
        and all(endpoint_beats_16.values())
    )
    print(f"g6d_capacity_trend_passed={str(passed).lower()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", required=True)
    p.add_argument("--validation-path", required=True)
    p.add_argument("--tokenizer-json", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threads", type=int, default=8)
    run(p.parse_args())
