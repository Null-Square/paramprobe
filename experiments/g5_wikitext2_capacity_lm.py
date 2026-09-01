"""G5: paired WikiText-2 raw capacity replication.

This is the second named-corpus replication of the frozen G3f protocol. It keeps
all model, router, page, optimization, and resource settings unchanged and only
changes the corpus to the official WikiText-2 raw train/validation split.

No Tiny Shakespeare hyperparameter is retuned here. The test remains byte-level
(256 symbols) so the architecture is identical to G3f.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

import g3a_internal_fixed_hash_lm as base
import g3b_prefix_reliable_lm as learned
import g3f_paired_capacity_lm as paired


def load_split(path: str) -> bytes:
    data = Path(path).read_bytes()
    if len(data) < 1024:
        raise ValueError(f"split is unexpectedly small: {path}")
    return data


def summarize(label: str, results: dict[int, list[float]]) -> None:
    print(label)
    print("pages\tvalidation_ce_mean\tvalidation_ce_sample_std")
    for used_bits, values_list in results.items():
        values = np.asarray(values_list, dtype=np.float64)
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        pages = 1 if used_bits == 0 else 2**used_bits
        print(f"{pages}\t{values.mean():.8f}\t{std:.8f}")


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    base.set_seed(args.backbone_seed)

    train_raw = load_split(args.train_path)
    validation_raw = load_split(args.validation_path)
    train = torch.tensor(list(train_raw), dtype=torch.long)
    validation = torch.tensor(list(validation_raw), dtype=torch.long)

    model = base.TwoBlockByteLM(
        d_model=args.d_model, heads=args.heads, context=args.context
    )
    checkpoint = Path(args.checkpoint)
    if args.retrain_backbone or not checkpoint.exists():
        base.pretrain_backbone(
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

    base_ce = base.evaluate_backbone(
        model, validation, args.eval_steps, args.eval_batch_size
    )
    projection, thresholds = base.build_balanced_hash(
        model,
        train,
        max_bits=args.max_bits,
        calibration_batches=args.hash_calibration_batches,
        batch_size=args.hash_batch_size,
    )
    router = learned.train_router(model, train, args)
    order, factor_stability = learned.reliability_order(model, router, train, args)

    used_values = [int(v) for v in args.used_bits.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    fixed_results: dict[int, list[float]] = {u: [] for u in used_values}
    learned_results: dict[int, list[float]] = {u: [] for u in used_values}
    fixed_by_seed: dict[int, list[float]] = {s: [] for s in seeds}
    learned_by_seed: dict[int, list[float]] = {s: [] for s in seeds}

    page_parameters = 10 * model.d_model + 10 + model.d_model * 10 + model.d_model
    page_payload_bytes = 4 * page_parameters
    active_page_macs = 2 * model.d_model * 10
    fixed_router_macs = model.d_model * args.max_bits
    learned_router_macs = (
        model.d_model * args.router_hidden_dim
        + args.router_hidden_dim * args.max_bits
    )

    print("# ParamProbe G5: paired WikiText-2 raw capacity replication")
    print(f"train_bytes={len(train_raw)}")
    print(f"validation_bytes={len(validation_raw)}")
    print(f"backbone_validation_ce={base_ce:.8f}")
    print(f"page_parameters={page_parameters}")
    print(f"page_payload_bytes={page_payload_bytes}")
    print(f"block_bytes={args.block_bytes}")
    print(f"active_page_macs_per_token={active_page_macs}")
    print("q_inference=1")
    print(f"logical_external_bytes_per_token={args.block_bytes}")
    print(f"fixed_router_macs_per_token={fixed_router_macs}")
    print(f"learned_router_macs_per_token={learned_router_macs}")
    print(f"page_init_seed_base={args.page_init_seed}")
    print(f"page_batch_seed_base={args.page_batch_seed}")
    print("paired_page_prefix_initialization=true")
    print("paired_training_minibatches=true")
    print("official_dataset_split=true")
    print(
        "factor_stability="
        + ",".join(f"{v:.8f}" for v in factor_stability.tolist())
    )
    print("reliability_order=" + ",".join(map(str, order.tolist())))
    print()
    print("seed\tused_bits\tpages\tfixed_ce\tlearned_ce")

    for seed in seeds:
        for used_bits in used_values:
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            fixed_pages = paired.train_fixed_pages(
                model,
                train,
                projection,
                thresholds,
                used_bits,
                args,
                seed,
            )
            fixed_ce = paired.evaluate_fixed(
                model,
                fixed_pages,
                validation,
                projection,
                thresholds,
                used_bits,
                args,
            )

            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            learned_pages = paired.train_learned_pages(
                model,
                router,
                train,
                used_bits,
                order,
                args,
                seed,
            )
            learned_ce = paired.evaluate_learned(
                model,
                router,
                learned_pages,
                validation,
                used_bits,
                order,
                args,
            )

            pages = 1 if used_bits == 0 else 2**used_bits
            fixed_results[used_bits].append(fixed_ce)
            learned_results[used_bits].append(learned_ce)
            fixed_by_seed[seed].append(fixed_ce)
            learned_by_seed[seed].append(learned_ce)
            print(
                f"{seed}\t{used_bits}\t{pages}\t{fixed_ce:.8f}\t{learned_ce:.8f}"
            )

    print()
    summarize("fixed_hash", fixed_results)
    print()
    summarize("learned_router", learned_results)

    fixed_monotone = {
        seed: bool(np.all(np.diff(np.asarray(values)) < 0.0))
        for seed, values in fixed_by_seed.items()
    }
    learned_monotone = {
        seed: bool(np.all(np.diff(np.asarray(values)) < 0.0))
        for seed, values in learned_by_seed.items()
    }
    print()
    print(
        "fixed_monotone_by_seed="
        + ",".join(
            f"{s}:{str(v).lower()}" for s, v in fixed_monotone.items()
        )
    )
    print(
        "learned_monotone_by_seed="
        + ",".join(
            f"{s}:{str(v).lower()}" for s, v in learned_monotone.items()
        )
    )
    print(
        "fixed_gate_passed="
        + str(len(seeds) >= 3 and all(fixed_monotone.values())).lower()
    )
    print(
        "learned_gate_passed="
        + str(len(seeds) >= 3 and all(learned_monotone.values())).lower()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--validation-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=".cache/paramprobe_g5_base.pt")
    parser.add_argument("--retrain-backbone", action="store_true")
    parser.add_argument("--backbone-seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--pretrain-batch-size", type=int, default=32)
    parser.add_argument("--pretrain-lr", type=float, default=3e-3)
    parser.add_argument("--max-bits", type=int, default=8)
    parser.add_argument("--used-bits", type=str, default="0,2,4,6,8")
    parser.add_argument("--hash-calibration-batches", type=int, default=50)
    parser.add_argument("--hash-batch-size", type=int, default=24)
    parser.add_argument("--router-hidden-dim", type=int, default=64)
    parser.add_argument("--router-hidden-batches", type=int, default=80)
    parser.add_argument("--router-hidden-batch-size", type=int, default=24)
    parser.add_argument("--router-hidden-seed", type=int, default=2468)
    parser.add_argument("--router-seed", type=int, default=17)
    parser.add_argument("--router-steps", type=int, default=600)
    parser.add_argument("--router-batch-size", type=int, default=1024)
    parser.add_argument("--router-lr", type=float, default=2e-3)
    parser.add_argument("--router-weight-decay", type=float, default=1e-4)
    parser.add_argument("--router-noise-std", type=float, default=0.08)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--balance-weight", type=float, default=0.2)
    parser.add_argument("--confidence-weight", type=float, default=0.1)
    parser.add_argument("--balance-prefixes", type=str, default="2,4,6,8")
    parser.add_argument("--order-hidden-batches", type=int, default=40)
    parser.add_argument("--order-hidden-batch-size", type=int, default=24)
    parser.add_argument("--order-hidden-seed", type=int, default=97531)
    parser.add_argument("--order-noise-seed", type=int, default=86420)
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--page-init-seed", type=int, default=40000)
    parser.add_argument("--page-batch-seed", type=int, default=50000)
    parser.add_argument("--seeds", type=str, default="7,8,9")
    parser.add_argument("--page-steps", type=int, default=180)
    parser.add_argument("--page-batch-size", type=int, default=24)
    parser.add_argument("--page-lr", type=float, default=4e-3)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())