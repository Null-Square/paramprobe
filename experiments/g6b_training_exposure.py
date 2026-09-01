"""G6b: predeclared training-exposure diagnostic for frozen G6a N=256.

The protocol is frozen in docs/g6b_training_exposure_predeclared.md before this
file was created.  G6b does not retry the G6a capacity gate.  It loads the exact
archived G6a tokenizer/backbone, keeps architecture/routing/inference resources
fixed, and extends only the N=256 page-training trajectory to 1,920 steps.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import g6a_subword_scale_lm as g6a


FROZEN_TOKENIZER_SHA256 = (
    "230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb"
)
FROZEN_BACKBONE_SHA256 = (
    "a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049"
)
FROZEN_N16_CE = {
    7: 5.70245363,
    8: 5.70234185,
    9: 5.70222968,
}
FROZEN_N256_120_CE = {
    7: 5.70341599,
    8: 5.70355248,
    9: 5.70339634,
}
CHECKPOINT_STEPS = (120, 240, 480, 960, 1920)


def sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_assets(args: argparse.Namespace) -> tuple[Tokenizer, g6a.SubwordCausalLM]:
    tokenizer_sha = sha256_path(args.tokenizer_json)
    backbone_sha = sha256_path(args.checkpoint)
    if tokenizer_sha != FROZEN_TOKENIZER_SHA256:
        raise RuntimeError(
            f"frozen tokenizer SHA mismatch: {tokenizer_sha} != {FROZEN_TOKENIZER_SHA256}"
        )
    if backbone_sha != FROZEN_BACKBONE_SHA256:
        raise RuntimeError(
            f"frozen backbone SHA mismatch: {backbone_sha} != {FROZEN_BACKBONE_SHA256}"
        )

    tokenizer = Tokenizer.from_file(args.tokenizer_json)
    if tokenizer.get_vocab_size() != 1024:
        raise RuntimeError("frozen tokenizer vocabulary is no longer 1024")

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
    return tokenizer, model


def route_stats(
    eval_bank: list[tuple[torch.Tensor, torch.Tensor]],
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    num_pages: int,
) -> tuple[float, float]:
    counts = torch.zeros(num_pages, dtype=torch.float64)
    with torch.no_grad():
        for hidden, _ in eval_bank:
            ids = g6a.route(hidden, projection, thresholds, 8)
            counts += torch.bincount(
                ids.reshape(-1).cpu(), minlength=num_pages
            ).double()
    raw = counts.numpy()
    positive = raw[raw > 0]
    probs = positive / positive.sum()
    entropy = float(-(probs * np.log(probs)).sum() / np.log(num_pages))
    dead = float(np.mean(raw == 0))
    return entropy, dead


def evaluate(
    model: g6a.SubwordCausalLM,
    pages: g6a.PageResidualMLP,
    eval_bank: list[tuple[torch.Tensor, torch.Tensor]],
    projection: torch.Tensor,
    thresholds: torch.Tensor,
) -> float:
    values: list[float] = []
    model.eval()
    pages.eval()
    with torch.no_grad():
        for hidden, y in eval_bank:
            ids = g6a.route(hidden, projection, thresholds, 8)
            logits = model.head(model.rest(hidden + pages(ids, hidden)))
            values.append(
                F.cross_entropy(
                    logits.reshape(-1, model.vocab_size), y.reshape(-1)
                ).item()
            )
    return float(np.mean(values))


def train_one_seed(
    model: g6a.SubwordCausalLM,
    train: torch.Tensor,
    eval_bank: list[tuple[torch.Tensor, torch.Tensor]],
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> dict[int, float]:
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    pages = g6a.PageResidualMLP(
        256,
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

    optimizer = torch.optim.AdamW(pages.parameters(), lr=4e-3, weight_decay=1e-4)
    torch.manual_seed(50000 + seed)
    pages.train()
    results: dict[int, float] = {}

    for step in range(1, CHECKPOINT_STEPS[-1] + 1):
        x, y = g6a.get_batch(train, 128, 8)
        with torch.no_grad():
            hidden = model.first(x)
            ids = g6a.route(hidden, projection, thresholds, 8)
        logits = model.head(model.rest(hidden + pages(ids, hidden)))
        loss = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in CHECKPOINT_STEPS:
            ce = evaluate(model, pages, eval_bank, projection, thresholds)
            results[step] = ce
            assignments_per_page = step * 8 * 128 / 256
            print(
                f"{seed}\t{step}\t{assignments_per_page:.1f}\t{ce:.8f}",
                flush=True,
            )
            pages.train()

    return results


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    tokenizer, model = load_frozen_assets(args)
    train = g6a.encode_split(tokenizer, args.train_path)
    validation = g6a.encode_split(tokenizer, args.validation_path)

    if len(train) != 4_254_523 or len(validation) != 445_470:
        raise RuntimeError(
            f"tokenized split sizes changed: train={len(train)} validation={len(validation)}"
        )

    projection, thresholds = g6a.build_balanced_hash(
        model,
        train,
        max_bits=8,
        calibration_batches=40,
        batch_size=8,
    )
    eval_bank = g6a.make_hidden_bank(
        model,
        validation,
        steps=40,
        batch_size=8,
        seed=1234,
    )
    entropy, dead = route_stats(eval_bank, projection, thresholds, 256)

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
        raise RuntimeError("frozen G6b inference resource envelope changed")

    print("# ParamProbe G6b: frozen N=256 training-exposure diagnostic")
    print("protocol_predeclared=true")
    print(f"tokenizer_sha256={sha256_path(args.tokenizer_json)}")
    print(f"backbone_sha256={sha256_path(args.checkpoint)}")
    print(f"train_tokens={len(train)}")
    print(f"validation_tokens={len(validation)}")
    print("num_pages=256")
    print("page_parameters=3956")
    print("page_payload_bytes=15824")
    print("block_bytes=16384")
    print("q_inference=1")
    print("logical_external_bytes_per_token=16384")
    print("active_page_macs_per_token=3840")
    print("fixed_router_macs_per_token=768")
    print(f"util_entropy={entropy:.8f}")
    print(f"dead_page_fraction={dead:.8f}")
    print("checkpoint_steps=" + ",".join(map(str, CHECKPOINT_STEPS)))
    print()
    print("seed\tsteps\tmean_assignments_per_page\tvalidation_ce")

    all_results: dict[int, dict[int, float]] = {}
    for seed in (7, 8, 9):
        all_results[seed] = train_one_seed(
            model,
            train,
            eval_bank,
            projection,
            thresholds,
            args,
            seed,
        )

    integrity: dict[int, bool] = {}
    beats_n16: dict[int, bool] = {}
    improves_from_120: dict[int, bool] = {}
    trajectory_monotone: dict[int, bool] = {}
    tolerance = 2e-7

    print()
    print(
        "seed\treplicated_120\tfrozen_g6a_120\tabs_error\tfrozen_n16_120\tfinal_1920\tbeats_n16"
    )
    for seed, values in all_results.items():
        replicated = values[120]
        frozen_256 = FROZEN_N256_120_CE[seed]
        frozen_16 = FROZEN_N16_CE[seed]
        final = values[1920]
        error = abs(replicated - frozen_256)
        integrity[seed] = error <= tolerance
        beats_n16[seed] = final < frozen_16
        improves_from_120[seed] = final < replicated
        sequence = np.asarray([values[s] for s in CHECKPOINT_STEPS])
        trajectory_monotone[seed] = bool(np.all(np.diff(sequence) < 0.0))
        print(
            f"{seed}\t{replicated:.8f}\t{frozen_256:.8f}\t{error:.10f}\t"
            f"{frozen_16:.8f}\t{final:.8f}\t{str(beats_n16[seed]).lower()}"
        )

    means = {
        step: float(np.mean([all_results[s][step] for s in (7, 8, 9)]))
        for step in CHECKPOINT_STEPS
    }
    stds = {
        step: float(np.std([all_results[s][step] for s in (7, 8, 9)], ddof=1))
        for step in CHECKPOINT_STEPS
    }
    print()
    print("steps\tmean_assignments_per_page\tvalidation_ce_mean\tvalidation_ce_sample_std")
    for step in CHECKPOINT_STEPS:
        assignments = step * 8 * 128 / 256
        print(f"{step}\t{assignments:.1f}\t{means[step]:.8f}\t{stds[step]:.8f}")

    print()
    print(
        "replication_integrity_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in integrity.items())
    )
    print(
        "beats_frozen_n16_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in beats_n16.items())
    )
    print(
        "improves_vs_120_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in improves_from_120.items())
    )
    print(
        "checkpoint_monotone_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in trajectory_monotone.items())
    )
    print("resource_gate_passed=true")
    print(
        "g6b_exposure_supported="
        + str(
            all(integrity.values())
            and all(beats_n16.values())
            and all(improves_from_120.values())
        ).lower()
    )

    if not all(integrity.values()):
        raise RuntimeError("G6b failed the predeclared G6a 120-step replication check")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())
