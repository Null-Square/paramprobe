"""G6e: paired robustness estimate for the frozen G6d N=4 versus N=16 edge.

Predeclared in docs/g6e_low_capacity_edge_predeclared.md before this file was
created.  This is an estimation study and cannot change the frozen G6d gate.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import g6a_subword_scale_lm as g6a
import g6d_capacity_scaled_training as g6d

SEEDS = tuple(range(16, 32))
T_CRIT_DF15 = 2.1314495456


def train_condition(
    model: g6a.SubwordCausalLM,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
    eval_bank: list[tuple[torch.Tensor, torch.Tensor]],
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    used_bits: int,
    seed: int,
) -> float:
    n = 2**used_bits
    pages = g6d.init_pages(n, seed)
    optimizer = torch.optim.AdamW(pages.parameters(), lr=4e-3, weight_decay=1e-4)
    pages.train()
    for hidden, y in bank:
        ids = g6a.route(hidden, projection, thresholds, used_bits)
        logits = model.head(model.rest(hidden + pages(ids, hidden)))
        loss = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return g6d.evaluate(
        model, pages, eval_bank, projection, thresholds, used_bits
    )


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    tokenizer, model = g6d.load_assets(args)
    train = g6a.encode_split(tokenizer, args.train_path)
    valid = g6a.encode_split(tokenizer, args.validation_path)
    if len(train) != 4_254_523 or len(valid) != 445_470:
        raise RuntimeError("tokenized split sizes changed")

    projection, thresholds = g6a.build_balanced_hash(
        model, train, max_bits=8, calibration_batches=40, batch_size=8
    )
    eval_bank = g6a.make_hidden_bank(model, valid, 40, 8, 1234)

    # Freeze the same inference envelope used by G6d.
    page_parameters = 20 * 96 + 20 + 96 * 20 + 96
    if (
        page_parameters != 3956
        or 4 * page_parameters != 15824
        or 2 * 96 * 20 != 3840
        or 96 * 8 != 768
    ):
        raise RuntimeError("frozen inference resource envelope changed")

    print("# ParamProbe G6e: paired N=4 vs N=16 robustness estimate")
    print("protocol_predeclared=true")
    print("fresh_page_seeds=16..31")
    print(f"tokenizer_sha256={g6d.sha256_path(args.tokenizer_json)}")
    print(f"backbone_sha256={g6d.sha256_path(args.checkpoint)}")
    print("q_inference=1")
    print("block_bytes=16384")
    print("logical_external_bytes_per_token=16384")
    print("page_parameters=3956")
    print("page_payload_bytes=15824")
    print("active_page_macs_per_token=3840")
    print("fixed_router_macs_per_token=768")
    print("training_steps_per_condition=120")
    print("total_routed_assignments_per_condition=122880")
    print("n4_mean_assignments_per_page=30720")
    print("n16_mean_assignments_per_page=7680")
    print()
    print("seed\tn4_ce\tn16_ce\tdelta_n16_minus_n4\tn16_better")

    deltas: list[float] = []
    n4_values: list[float] = []
    n16_values: list[float] = []
    for seed in SEEDS:
        # Exactly paired hidden-state/target minibatches across capacities.
        bank = g6a.make_hidden_bank(model, train, 120, 8, 50000 + seed)
        ce4 = train_condition(
            model, bank, eval_bank, projection, thresholds, 2, seed
        )
        ce16 = train_condition(
            model, bank, eval_bank, projection, thresholds, 4, seed
        )
        delta = ce16 - ce4
        n4_values.append(ce4)
        n16_values.append(ce16)
        deltas.append(delta)
        print(
            f"{seed}\t{ce4:.8f}\t{ce16:.8f}\t{delta:+.8f}\t"
            f"{str(delta < 0.0).lower()}",
            flush=True,
        )

    arr = np.asarray(deltas, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    se = std / math.sqrt(len(arr))
    half = T_CRIT_DF15 * se
    lo, hi = mean - half, mean + half
    n16_better_count = int(np.sum(arr < 0.0))

    print()
    print(f"n4_ce_mean={np.mean(n4_values):.8f}")
    print(f"n16_ce_mean={np.mean(n16_values):.8f}")
    print(f"paired_delta_mean={mean:+.8f}")
    print(f"paired_delta_sample_std={std:.8f}")
    print(f"paired_delta_standard_error={se:.8f}")
    print(f"paired_delta_95pct_t_interval=[{lo:+.8f},{hi:+.8f}]")
    print(f"n16_better_seed_count={n16_better_count}/{len(SEEDS)}")
    if hi < 0.0:
        classification = "supports_n16"
    elif lo > 0.0:
        classification = "supports_n4"
    else:
        classification = "unresolved"
    print(f"g6e_edge_classification={classification}")
    print("resource_gate_passed=true")
    print("g6d_gate_reclassified=false")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", required=True)
    p.add_argument("--validation-path", required=True)
    p.add_argument("--tokenizer-json", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threads", type=int, default=8)
    run(p.parse_args())
