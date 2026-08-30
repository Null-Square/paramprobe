"""G3e: Tiny Shakespeare page-size / operator-granularity sweep.

This experiment freezes the G3c fixed-hash routing mechanism and asks a different
question from the page-count scaling curve: at fixed *total* external parameter
capacity, how does changing physical page size alter active traffic, local
operator capacity, and held-out language-model loss?

The default conditions all expose exactly 1 MiB of external parameter storage:

- 256 pages x 4 KiB;
- 64 pages x 16 KiB;
- 16 pages x 64 KiB.

The maximum eight-factor fixed hash is allocated and executed for every
condition.  Smaller page counts use prefixes of that same hash, so resident
router metadata and routing MACs are fixed.  Each physical page is filled by the
largest 48 -> h -> 48 FP32 residual MLP whose complete parameter payload fits in
the page.  Consequently both bytes read per token and active page compute grow
with B; this is an operator-granularity experiment, not a fixed-active-compute
capacity sweep.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import g3a_internal_fixed_hash_lm as base


def max_hidden_dim_for_block(block_bytes: int, d_model: int) -> int:
    # parameters(h) = h*d + h + d*h + d = h*(2d+1) + d
    available_parameters = block_bytes // 4
    hidden = (available_parameters - d_model) // (2 * d_model + 1)
    if hidden < 1:
        raise ValueError("block is too small for even a width-1 page MLP")
    return int(hidden)


def parse_conditions(spec: str) -> list[tuple[int, int]]:
    values: list[tuple[int, int]] = []
    for item in spec.split(","):
        block_text, bits_text = item.split(":", 1)
        block_bytes = int(block_text)
        used_bits = int(bits_text)
        if block_bytes <= 0 or used_bits < 0:
            raise ValueError("invalid page-size condition")
        values.append((block_bytes, used_bits))
    if not values:
        raise ValueError("at least one condition is required")
    return values


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
    base.set_seed(seed)
    pages = base.PageResidualMLP(
        num_pages,
        d_model=model.d_model,
        hidden_dim=hidden_dim,
        residual_scale=0.15,
        block_bytes=block_bytes,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        pages.parameters(), lr=args.page_lr, weight_decay=1e-4
    )
    model.eval()
    pages.train()
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


def normalized_entropy(counts: torch.Tensor) -> float:
    raw = counts.double().numpy()
    positive = raw[raw > 0]
    if len(raw) == 1:
        return 1.0
    probabilities = positive / positive.sum()
    return float(
        -(probabilities * np.log(probabilities)).sum() / np.log(len(raw))
    )


def evaluate_pages(
    model: base.TwoBlockByteLM,
    pages: base.PageResidualMLP,
    validation: torch.Tensor,
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    used_bits: int,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    torch.manual_seed(args.eval_seed)
    values: list[float] = []
    counts = torch.zeros(pages.num_pages)
    model.eval()
    pages.eval()
    with torch.no_grad():
        for _ in range(args.eval_steps):
            x, y = base.get_batch(
                validation, model.context, args.eval_batch_size
            )
            hidden = model.first(x)
            page_ids = base.route(
                hidden, projection, thresholds, used_bits
            )
            logits = model.head(model.rest(hidden + pages(page_ids, hidden)))
            values.append(
                F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).item()
            )
            counts += torch.bincount(
                page_ids.reshape(-1).cpu(), minlength=pages.num_pages
            )
    return (
        float(np.mean(values)),
        normalized_entropy(counts),
        float((counts == 0).double().mean().item()),
    )


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
    router_scalars = model.d_model * args.max_bits + args.max_bits
    router_macs = model.d_model * args.max_bits

    conditions = parse_conditions(args.conditions)
    target_capacity: int | None = None
    for block_bytes, used_bits in conditions:
        pages = 1 if used_bits == 0 else 2**used_bits
        capacity = pages * block_bytes
        if target_capacity is None:
            target_capacity = capacity
        elif capacity != target_capacity:
            raise ValueError(
                "all default G3e conditions must hold total external bytes fixed"
            )

    print("# ParamProbe G3e: Tiny Shakespeare page granularity")
    print(f"corpus_bytes={len(raw)}")
    print(f"backbone_validation_ce={base_ce:.8f}")
    print(f"max_hash_bits={args.max_bits}")
    print(f"fixed_router_scalars={router_scalars}")
    print(f"fixed_routing_macs_per_token={router_macs}")
    print(f"fixed_total_external_bytes={target_capacity}")
    print("q_inference=1")
    print("page_hidden_width=max_width_that_fits_complete_fp32_page")
    print("residual_scale=0.15")
    print()
    print(
        "seed\tblock_bytes\tused_bits\tpages\thidden_dim\t"
        "page_parameters\tpayload_bytes\tactive_page_macs\t"
        "validation_ce\tutil_entropy\tdead_pages"
    )

    seeds = [int(value) for value in args.seeds.split(",")]
    results: dict[tuple[int, int], list[float]] = {
        condition: [] for condition in conditions
    }
    for seed in seeds:
        for block_bytes, used_bits in conditions:
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            hidden_dim = max_hidden_dim_for_block(
                block_bytes, model.d_model
            )
            num_pages = 1 if used_bits == 0 else 2**used_bits
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
            ce, entropy, dead = evaluate_pages(
                model,
                pages,
                validation,
                projection,
                thresholds,
                used_bits,
                args,
            )
            active_macs = 2 * model.d_model * hidden_dim
            results[(block_bytes, used_bits)].append(ce)
            print(
                f"{seed}\t{block_bytes}\t{used_bits}\t{num_pages}\t"
                f"{hidden_dim}\t{pages.page_parameters}\t"
                f"{pages.page_payload_bytes}\t{active_macs}\t"
                f"{ce:.8f}\t{entropy:.8f}\t{dead:.8f}"
            )

    print()
    print(
        "block_bytes\tpages\thidden_dim\tlogical_bytes_per_token\t"
        "active_page_macs\tvalidation_ce_mean\tvalidation_ce_sample_std"
    )
    for block_bytes, used_bits in conditions:
        values = np.asarray(
            results[(block_bytes, used_bits)], dtype=np.float64
        )
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        num_pages = 1 if used_bits == 0 else 2**used_bits
        hidden_dim = max_hidden_dim_for_block(block_bytes, model.d_model)
        active_macs = 2 * model.d_model * hidden_dim
        print(
            f"{block_bytes}\t{num_pages}\t{hidden_dim}\t{block_bytes}\t"
            f"{active_macs}\t{values.mean():.8f}\t{std:.8f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-path", type=str, default=None)
    parser.add_argument("--corpus-bytes", type=int, default=1_200_000)
    parser.add_argument(
        "--checkpoint", type=str, default=".cache/paramprobe_g3e_base.pt"
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
        "--conditions",
        type=str,
        default="4096:8,16384:6,65536:4",
        help="comma-separated block_bytes:used_bits pairs",
    )
    parser.add_argument("--seeds", type=str, default="7,8,9")
    parser.add_argument("--page-steps", type=int, default=180)
    parser.add_argument("--page-batch-size", type=int, default=24)
    parser.add_argument("--page-lr", type=float, default=4e-3)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())