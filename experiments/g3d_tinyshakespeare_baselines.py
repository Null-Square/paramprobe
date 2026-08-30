"""G3d: matched Tiny Shakespeare baselines for the ParamProbe LM sweep.

This file is deliberately a baseline experiment rather than a new ParamProbe
variant.  It uses the same frozen two-block byte LM and internal insertion point
as G3c and compares two conventional controls:

1. a resident dense adapter with exactly the same 48 -> 10 -> 48 residual MLP
   shape and active matrix MAC count as one ParamProbe page;
2. a flat sparse/MoE-style control with a resident 256-way linear router,
   top-2 expert execution during training, and top-1 execution at inference.

The flat router always allocates and executes all 256 logits, even when only a
prefix of 1/4/16/64/256 experts is active, so its resident router parameters and
routing MACs are fixed across this finite sweep.  Unlike ParamProbe's factorized
router, however, this is O(N_max) resident routing metadata and is therefore a
comparison baseline, not part of the asymptotic claim.

No semantic expert labels or future-token routing targets are used.  The sparse
router is trained task-only from causal hidden states through the selected
mixture plus a small full-softmax load-balance penalty.  Inference evaluates
exactly one expert, but the baseline makes no external-I/O claim because all
experts are treated as resident conventional MoE state.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import g3a_internal_fixed_hash_lm as base


class FlatTop2MoE(nn.Module):
    """Maximum-width flat router plus page-shaped resident experts."""

    def __init__(
        self,
        num_experts: int,
        max_experts: int,
        d_model: int,
        hidden_dim: int,
        block_bytes: int,
        page_seed: int,
        router_seed: int,
    ) -> None:
        super().__init__()
        if not 1 <= num_experts <= max_experts:
            raise ValueError("num_experts must lie in [1, max_experts]")
        self.num_experts = num_experts
        self.max_experts = max_experts

        # Keep expert initialization controlled independently from the router.
        base.set_seed(page_seed)
        self.experts = base.PageResidualMLP(
            num_experts,
            d_model=d_model,
            hidden_dim=hidden_dim,
            residual_scale=0.15,
            block_bytes=block_bytes,
        )
        torch.manual_seed(router_seed)
        self.router = nn.Linear(d_model, max_experts, bias=True)

    def active_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        # The maximum-width projection is always executed.  Slicing happens only
        # after all max_experts logits have been computed.
        return self.router(hidden)[..., : self.num_experts]

    def training_residual(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.active_logits(hidden)
        probabilities = torch.softmax(logits, dim=-1)

        if self.num_experts == 1:
            ids = torch.zeros(
                hidden.shape[:-1], dtype=torch.long, device=hidden.device
            )
            return self.experts(ids, hidden), hidden.new_zeros(())

        top_values, top_ids = torch.topk(logits, k=2, dim=-1)
        top_weights = torch.softmax(top_values, dim=-1)
        expanded = hidden.unsqueeze(-2).expand(
            *hidden.shape[:-1], 2, hidden.shape[-1]
        )
        expert_values = self.experts(top_ids, expanded)
        residual = (top_weights.unsqueeze(-1) * expert_values).sum(dim=-2)

        # Switch-style importance regularization over the dense router
        # probabilities.  It is exactly zero for uniform mean importance.
        mean_probability = probabilities.reshape(-1, self.num_experts).mean(dim=0)
        balance = (
            self.num_experts * torch.sum(mean_probability.square()) - 1.0
        )
        return residual, balance

    def inference_residual(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.active_logits(hidden)
        ids = torch.argmax(logits, dim=-1)
        return self.experts(ids, hidden), ids


def load_problem(
    args: argparse.Namespace,
) -> tuple[base.TwoBlockByteLM, torch.Tensor, torch.Tensor, bytes]:
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
    return model, train, validation, raw


def train_dense_adapter(
    model: base.TwoBlockByteLM,
    train: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> base.PageResidualMLP:
    base.set_seed(seed)
    adapter = base.PageResidualMLP(
        1,
        d_model=model.d_model,
        hidden_dim=args.expert_hidden_dim,
        residual_scale=0.15,
        block_bytes=args.block_bytes,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.page_lr, weight_decay=1e-4
    )
    model.eval()
    adapter.train()
    for _ in range(args.page_steps):
        x, y = base.get_batch(train, model.context, args.page_batch_size)
        with torch.no_grad():
            hidden = model.first(x)
        ids = torch.zeros(hidden.shape[:-1], dtype=torch.long)
        logits = model.head(model.rest(hidden + adapter(ids, hidden)))
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return adapter


def evaluate_dense_adapter(
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
            x, y = base.get_batch(validation, model.context, args.eval_batch_size)
            hidden = model.first(x)
            ids = torch.zeros(hidden.shape[:-1], dtype=torch.long)
            logits = model.head(model.rest(hidden + adapter(ids, hidden)))
            values.append(
                F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).item()
            )
    return float(np.mean(values))


def train_sparse_moe(
    model: base.TwoBlockByteLM,
    train: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
    num_experts: int,
) -> FlatTop2MoE:
    moe = FlatTop2MoE(
        num_experts=num_experts,
        max_experts=args.max_experts,
        d_model=model.d_model,
        hidden_dim=args.expert_hidden_dim,
        block_bytes=args.block_bytes,
        page_seed=seed,
        router_seed=args.router_seed + seed,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        moe.parameters(), lr=args.page_lr, weight_decay=1e-4
    )
    model.eval()
    moe.train()
    # Fix the data-sampling stream independently of expert/router initialization.
    torch.manual_seed(args.train_batch_seed + seed)
    for _ in range(args.page_steps):
        x, y = base.get_batch(train, model.context, args.page_batch_size)
        with torch.no_grad():
            hidden = model.first(x)
        residual, balance = moe.training_residual(hidden)
        logits = model.head(model.rest(hidden + residual))
        language_loss = F.cross_entropy(
            logits.reshape(-1, 256), y.reshape(-1)
        )
        loss = language_loss + args.balance_weight * balance
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(moe.router.parameters(), 1.0)
        optimizer.step()
    return moe


def normalized_entropy(counts: torch.Tensor) -> float:
    raw = counts.double().numpy()
    positive = raw[raw > 0]
    if len(raw) == 1:
        return 1.0
    probabilities = positive / positive.sum()
    return float(
        -(probabilities * np.log(probabilities)).sum() / np.log(len(raw))
    )


def evaluate_sparse_moe(
    model: base.TwoBlockByteLM,
    moe: FlatTop2MoE,
    validation: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    torch.manual_seed(args.eval_seed)
    values: list[float] = []
    counts = torch.zeros(moe.num_experts)
    model.eval()
    moe.eval()
    with torch.no_grad():
        for _ in range(args.eval_steps):
            x, y = base.get_batch(validation, model.context, args.eval_batch_size)
            hidden = model.first(x)
            residual, ids = moe.inference_residual(hidden)
            logits = model.head(model.rest(hidden + residual))
            values.append(
                F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).item()
            )
            counts += torch.bincount(
                ids.reshape(-1).cpu(), minlength=moe.num_experts
            )
    return (
        float(np.mean(values)),
        normalized_entropy(counts),
        float((counts == 0).double().mean().item()),
    )


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    model, train, validation, raw = load_problem(args)
    checkpoint = Path(args.checkpoint)
    base_ce = base.evaluate_backbone(
        model, validation, args.eval_steps, args.eval_batch_size
    )

    expert_parameters = (
        args.expert_hidden_dim * model.d_model
        + args.expert_hidden_dim
        + model.d_model * args.expert_hidden_dim
        + model.d_model
    )
    expert_payload_bytes = 4 * expert_parameters
    active_expert_macs = 2 * model.d_model * args.expert_hidden_dim
    flat_router_parameters = (
        model.d_model * args.max_experts + args.max_experts
    )
    flat_router_macs = model.d_model * args.max_experts

    print("# ParamProbe G3d: Tiny Shakespeare matched baselines")
    print(f"corpus_bytes={len(raw)}")
    print(f"backbone_validation_ce={base_ce:.8f}")
    print(f"expert_parameters={expert_parameters}")
    print(f"expert_payload_bytes={expert_payload_bytes}")
    print(f"active_expert_matrix_macs_per_token={active_expert_macs}")
    print(f"block_bytes_reference={args.block_bytes}")
    print(f"flat_router_max_experts={args.max_experts}")
    print(f"flat_router_parameters={flat_router_parameters}")
    print(f"flat_router_matrix_macs_per_token={flat_router_macs}")
    print("sparse_training_experts_per_token=2_when_N_gt_1")
    print("sparse_inference_experts_per_token=1")
    print("sparse_router_supervision=task_only_causal_hidden_state")
    print(f"sparse_balance_weight={args.balance_weight}")
    print("baseline_external_io_claim=none")
    print()

    seeds = [int(value) for value in args.seeds.split(",")]
    expert_counts = [int(value) for value in args.expert_counts.split(",")]

    print("dense_adapter_seed\tvalidation_ce")
    dense_results: list[float] = []
    for seed in seeds:
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        adapter = train_dense_adapter(model, train, args, seed)
        ce = evaluate_dense_adapter(model, adapter, validation, args)
        dense_results.append(ce)
        print(f"{seed}\t{ce:.8f}")
    dense_array = np.asarray(dense_results, dtype=np.float64)
    dense_std = float(dense_array.std(ddof=1)) if len(dense_array) > 1 else float("nan")
    print(
        f"dense_adapter_mean={dense_array.mean():.8f}\t"
        f"sample_std={dense_std:.8f}"
    )
    print()

    print("seed\texperts\tvalidation_ce\tutil_entropy\tdead_experts")
    results: dict[int, list[float]] = {count: [] for count in expert_counts}
    for seed in seeds:
        for count in expert_counts:
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            moe = train_sparse_moe(model, train, args, seed, count)
            ce, entropy, dead = evaluate_sparse_moe(
                model, moe, validation, args
            )
            results[count].append(ce)
            print(
                f"{seed}\t{count}\t{ce:.8f}\t{entropy:.8f}\t{dead:.8f}"
            )

    print()
    print("experts\tvalidation_ce_mean\tvalidation_ce_sample_std")
    for count in expert_counts:
        values = np.asarray(results[count], dtype=np.float64)
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        print(f"{count}\t{values.mean():.8f}\t{std:.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-path", type=str, default=None)
    parser.add_argument("--corpus-bytes", type=int, default=1_200_000)
    parser.add_argument(
        "--checkpoint", type=str, default=".cache/paramprobe_g3d_base.pt"
    )
    parser.add_argument("--retrain-backbone", action="store_true")
    parser.add_argument("--backbone-seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--pretrain-batch-size", type=int, default=32)
    parser.add_argument("--pretrain-lr", type=float, default=3e-3)
    parser.add_argument("--expert-hidden-dim", type=int, default=10)
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--max-experts", type=int, default=256)
    parser.add_argument("--expert-counts", type=str, default="1,4,16,64,256")
    parser.add_argument("--router-seed", type=int, default=20000)
    parser.add_argument("--train-batch-seed", type=int, default=30000)
    parser.add_argument("--balance-weight", type=float, default=0.01)
    parser.add_argument("--seeds", type=str, default="7,8,9")
    parser.add_argument("--page-steps", type=int, default=180)
    parser.add_argument("--page-batch-size", type=int, default=24)
    parser.add_argument("--page-lr", type=float, default=4e-3)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())