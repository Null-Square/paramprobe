"""G3g: resident dense-adapter compute baselines on Tiny Shakespeare.

Two resident adapters are compared at the successful ParamProbe insertion point:

- width 10: exactly matches the active 48 -> 10 -> 48 ParamProbe page compute
  (960 matrix MACs/token);
- width 47: approximately matches the *total* learned ParamProbe matrix compute
  at G3c max width: 4512 dense-adapter MACs/token versus
  3584 router + 960 page = 4544 ParamProbe MACs/token.

Both baselines are fully resident and make no external-I/O claim.  They use the
same frozen backbone, optimizer, training steps, paired minibatch stream, and
evaluation batches.  The purpose is to test whether the G3c gain can be matched
simply by spending comparable active dense compute without inactive external
capacity.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import g3a_internal_fixed_hash_lm as base


def train_adapter(
    model: base.TwoBlockByteLM,
    train: torch.Tensor,
    hidden_dim: int,
    args: argparse.Namespace,
    seed: int,
) -> base.PageResidualMLP:
    # This is resident, so use a synthetic block bound large enough to avoid
    # conflating the dense baseline with ParamProbe's 4 KiB page constraint.
    parameters = hidden_dim * (2 * model.d_model + 1) + model.d_model
    payload = 4 * parameters
    base.set_seed(args.adapter_init_seed + seed)
    adapter = base.PageResidualMLP(
        1,
        d_model=model.d_model,
        hidden_dim=hidden_dim,
        residual_scale=0.15,
        block_bytes=payload,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    model.eval()
    adapter.train()
    torch.manual_seed(args.batch_seed + seed)
    for _ in range(args.steps):
        x, y = base.get_batch(train, model.context, args.batch_size)
        with torch.no_grad():
            hidden = model.first(x)
        ids = torch.zeros(hidden.shape[:-1], dtype=torch.long)
        logits = model.head(model.rest(hidden + adapter(ids, hidden)))
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return adapter


def evaluate(
    model: base.TwoBlockByteLM,
    adapter: base.PageResidualMLP,
    validation: torch.Tensor,
    args: argparse.Namespace,
) -> float:
    torch.manual_seed(args.eval_seed)
    values: list[float] = []
    model.eval()
    adapter.eval()
    with torch.no_grad():
        for _ in range(args.eval_steps):
            x, y = base.get_batch(
                validation, model.context, args.eval_batch_size
            )
            hidden = model.first(x)
            ids = torch.zeros(hidden.shape[:-1], dtype=torch.long)
            logits = model.head(model.rest(hidden + adapter(ids, hidden)))
            values.append(
                F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).item()
            )
    return float(np.mean(values))


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
    widths = [int(v) for v in args.widths.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]

    print("# ParamProbe G3g: resident dense compute baselines")
    print(f"corpus_bytes={len(raw)}")
    print(f"backbone_validation_ce={base_ce:.8f}")
    print(f"learned_paramprobe_router_macs={args.paramprobe_router_macs}")
    print(f"learned_paramprobe_page_macs={args.paramprobe_page_macs}")
    print(
        "learned_paramprobe_total_matrix_macs="
        f"{args.paramprobe_router_macs + args.paramprobe_page_macs}"
    )
    print("external_io_claim=none")
    print("paired_training_minibatches=true")
    print()
    print("seed\thidden_dim\tparameters\tpayload_bytes\tmatrix_macs\tvalidation_ce")

    results: dict[int, list[float]] = {width: [] for width in widths}
    for seed in seeds:
        for width in widths:
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            adapter = train_adapter(model, train, width, args, seed)
            ce = evaluate(model, adapter, validation, args)
            parameters = width * (2 * model.d_model + 1) + model.d_model
            payload = 4 * parameters
            macs = 2 * model.d_model * width
            results[width].append(ce)
            print(
                f"{seed}\t{width}\t{parameters}\t{payload}\t{macs}\t{ce:.8f}"
            )

    print()
    print("hidden_dim\tmatrix_macs\tvalidation_ce_mean\tvalidation_ce_sample_std")
    for width in widths:
        values = np.asarray(results[width], dtype=np.float64)
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        print(
            f"{width}\t{2 * model.d_model * width}\t"
            f"{values.mean():.8f}\t{std:.8f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-path", type=str, default=None)
    parser.add_argument("--corpus-bytes", type=int, default=1_200_000)
    parser.add_argument(
        "--checkpoint", type=str, default=".cache/paramprobe_g3g_base.pt"
    )
    parser.add_argument("--retrain-backbone", action="store_true")
    parser.add_argument("--backbone-seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--pretrain-batch-size", type=int, default=32)
    parser.add_argument("--pretrain-lr", type=float, default=3e-3)
    parser.add_argument("--widths", type=str, default="10,47")
    parser.add_argument("--paramprobe-router-macs", type=int, default=3584)
    parser.add_argument("--paramprobe-page-macs", type=int, default=960)
    parser.add_argument("--adapter-init-seed", type=int, default=60000)
    parser.add_argument("--batch-seed", type=int, default=70000)
    parser.add_argument("--seeds", type=str, default="7,8,9")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=4e-3)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())