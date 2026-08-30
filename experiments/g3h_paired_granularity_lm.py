"""G3h: paired replication of the Tiny Shakespeare page-granularity sweep.

This repeats G3e's fixed-total-capacity conditions while removing RNG coupling
between differently shaped page modules and the training minibatch stream.
Each condition gets a dedicated post-construction page initialization seed and
then the same minibatch sequence for a given replication seed.

Default conditions keep total external capacity at exactly 1 MiB:
256 x 4 KiB, 64 x 16 KiB, and 16 x 64 KiB.  The fixed maximum-width eight-bit
hash, q=1, optimizer, steps, residual scale, and evaluation batches are shared.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import g3a_internal_fixed_hash_lm as base
import g3e_page_granularity_lm as prior


def train_pages(
    model: base.TwoBlockByteLM,
    train: torch.Tensor,
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    used_bits: int,
    block_bytes: int,
    hidden_dim: int,
    args: argparse.Namespace,
    seed: int,
) -> base.PageResidualMLP:
    num_pages = 1 if used_bits == 0 else 2**used_bits
    pages = base.PageResidualMLP(
        num_pages,
        d_model=model.d_model,
        hidden_dim=hidden_dim,
        residual_scale=0.15,
        block_bytes=block_bytes,
    )
    # Remove constructor-shape-dependent RNG state from the learned init.
    torch.manual_seed(args.page_init_seed + seed)
    torch.nn.init.normal_(
        pages.w1.weight, std=0.15 / math.sqrt(model.d_model)
    )
    torch.nn.init.zeros_(pages.b1.weight)
    torch.nn.init.zeros_(pages.w2.weight)
    torch.nn.init.zeros_(pages.b2.weight)

    for parameter in model.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        pages.parameters(), lr=args.page_lr, weight_decay=1e-4
    )
    model.eval()
    pages.train()
    # Pair the sampled training examples across all B conditions.
    torch.manual_seed(args.page_batch_seed + seed)
    for _ in range(args.page_steps):
        x, y = base.get_batch(train, model.context, args.page_batch_size)
        with torch.no_grad():
            hidden = model.first(x)
            page_ids = base.route(
                hidden, projection, thresholds, used_bits
            )
        logits = model.head(model.rest(hidden + pages(page_ids, hidden)))
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return pages


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    base.set_seed(args.backbone_seed)
    raw = base.load_corpus(args.text_path, args.corpus_bytes)
    data = torch.tensor(list(raw), dtype=torch.long)
    split = int(0.9 * len(data))
    train, validation = data[:split], data[split:]

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
    conditions = prior.parse_conditions(args.conditions)
    capacities = {
        (1 if bits == 0 else 2**bits) * block for block, bits in conditions
    }
    if len(capacities) != 1:
        raise ValueError("conditions must hold total external bytes fixed")
    total_capacity = next(iter(capacities))

    print("# ParamProbe G3h: paired Tiny Shakespeare page granularity")
    print(f"corpus_bytes={len(raw)}")
    print(f"backbone_validation_ce={base_ce:.8f}")
    print(f"fixed_total_external_bytes={total_capacity}")
    print(f"fixed_routing_macs_per_token={model.d_model * args.max_bits}")
    print("q_inference=1")
    print(f"page_init_seed_base={args.page_init_seed}")
    print(f"page_batch_seed_base={args.page_batch_seed}")
    print("paired_training_minibatches=true")
    print()
    print(
        "seed\tblock_bytes\tpages\thidden_dim\tpayload_bytes\t"
        "active_page_macs\tvalidation_ce\tutil_entropy\tdead_pages"
    )

    seeds = [int(v) for v in args.seeds.split(",")]
    results: dict[tuple[int, int], list[float]] = {
        condition: [] for condition in conditions
    }
    for seed in seeds:
        for block_bytes, used_bits in conditions:
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            hidden_dim = prior.max_hidden_dim_for_block(
                block_bytes, model.d_model
            )
            pages = train_pages(
                model,
                train,
                projection,
                thresholds,
                used_bits,
                block_bytes,
                hidden_dim,
                args,
                seed,
            )
            ce, entropy, dead = prior.evaluate_pages(
                model,
                pages,
                validation,
                projection,
                thresholds,
                used_bits,
                args,
            )
            num_pages = 1 if used_bits == 0 else 2**used_bits
            active_macs = 2 * model.d_model * hidden_dim
            results[(block_bytes, used_bits)].append(ce)
            print(
                f"{seed}\t{block_bytes}\t{num_pages}\t{hidden_dim}\t"
                f"{pages.page_payload_bytes}\t{active_macs}\t{ce:.8f}\t"
                f"{entropy:.8f}\t{dead:.8f}"
            )

    print()
    print(
        "block_bytes\tpages\thidden_dim\tlogical_bytes_per_token\t"
        "active_page_macs\tvalidation_ce_mean\tvalidation_ce_sample_std"
    )
    for block_bytes, used_bits in conditions:
        values = np.asarray(results[(block_bytes, used_bits)], dtype=np.float64)
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        num_pages = 1 if used_bits == 0 else 2**used_bits
        hidden_dim = prior.max_hidden_dim_for_block(block_bytes, model.d_model)
        print(
            f"{block_bytes}\t{num_pages}\t{hidden_dim}\t{block_bytes}\t"
            f"{2 * model.d_model * hidden_dim}\t{values.mean():.8f}\t{std:.8f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-path", type=str, default=None)
    parser.add_argument("--corpus-bytes", type=int, default=1_200_000)
    parser.add_argument(
        "--checkpoint", type=str, default=".cache/paramprobe_g3h_base.pt"
    )
    parser.add_argument("--retrain-backbone", action="store_true")
    parser.add_argument("--backbone-seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--pretrain-batch-size", type=int, default=32)
    parser.add_argument("--pretrain-lr", type=float, default=3e-3)
    parser.add_argument("--max-bits", type=int, default=8)
    parser.add_argument("--hash-calibration-batches", type=int, default=50)
    parser.add_argument("--hash-batch-size", type=int, default=24)
    parser.add_argument(
        "--conditions", type=str, default="4096:8,16384:6,65536:4"
    )
    parser.add_argument("--page-init-seed", type=int, default=80000)
    parser.add_argument("--page-batch-seed", type=int, default=90000)
    parser.add_argument("--seeds", type=str, default="7,8,9")
    parser.add_argument("--page-steps", type=int, default=180)
    parser.add_argument("--page-batch-size", type=int, default=24)
    parser.add_argument("--page-lr", type=float, default=4e-3)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())