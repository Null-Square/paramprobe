"""G3b: learned causal factorized routing for an internal ParamProbe LM layer.

The router is trained without next-token labels or page-address supervision.
It sees only causal block-1 hidden states. Two perturbed views of the same hidden
state are encouraged to agree, the full composite address distribution is
balanced with the existing Renyi-2 collision objective, and factor probabilities
are pushed away from 0.5. The router is then frozen.

All capacity conditions reuse the same maximum-width router and take prefixes of
its six hard factors. Only the external page table size changes. Inference reads
exactly one 4096-byte page per token.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import g3a_internal_fixed_hash_lm as base
from paramprobe.routing_regularizers import (
    paired_collision_estimate,
    renyi2_address_deficit,
)


class LearnedCausalHash(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, max_bits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_bits = max_bits
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_bits),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


def bits_to_ids(bits: torch.Tensor, used_bits: int) -> torch.Tensor:
    if used_bits == 0:
        return torch.zeros(bits.shape[:-1], dtype=torch.long, device=bits.device)
    ids = torch.zeros(bits.shape[:-1], dtype=torch.long, device=bits.device)
    for factor in range(used_bits):
        ids = (ids << 1) | bits[..., factor].long()
    return ids


def route(
    router: LearnedCausalHash, hidden: torch.Tensor, used_bits: int
) -> torch.Tensor:
    # Always execute the full fixed-width router, even for smaller capacities.
    logits = router(hidden)
    return bits_to_ids(logits > 0, used_bits)


def collect_hidden_bank(
    model: base.TwoBlockByteLM,
    train: torch.Tensor,
    batches: int,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    values: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            x, _ = base.get_batch(train, model.context, batch_size)
            values.append(model.first(x).reshape(-1, model.d_model).cpu())
    return torch.cat(values, dim=0)


def train_router(
    model: base.TwoBlockByteLM,
    train: torch.Tensor,
    args: argparse.Namespace,
) -> LearnedCausalHash:
    hidden = collect_hidden_bank(
        model,
        train,
        args.router_hidden_batches,
        args.router_hidden_batch_size,
        args.router_hidden_seed,
    )
    feature_scale = hidden.std(dim=0, unbiased=False).clamp_min(1e-4)

    torch.manual_seed(args.router_seed)
    router = LearnedCausalHash(
        model.d_model, args.router_hidden_dim, args.max_bits
    )
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=args.router_lr,
        weight_decay=args.router_weight_decay,
    )
    router.train()

    for _ in range(args.router_steps):
        idx = torch.randint(0, len(hidden), (args.router_batch_size,))
        clean = hidden[idx]
        first = clean + args.router_noise_std * feature_scale * torch.randn_like(clean)
        second = clean + args.router_noise_std * feature_scale * torch.randn_like(clean)

        p_first = torch.sigmoid(router(first))
        p_second = torch.sigmoid(router(second))
        pooled = 0.5 * (p_first + p_second)

        consistency = 4.0 * torch.mean((p_first - p_second) ** 2)
        collision = paired_collision_estimate(pooled)
        balance = renyi2_address_deficit(collision, args.max_bits)
        confidence = torch.mean(4.0 * pooled * (1.0 - pooled))

        loss = (
            args.consistency_weight * consistency
            + args.balance_weight * balance
            + args.confidence_weight * confidence
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for parameter in router.parameters():
        parameter.requires_grad = False
    router.eval()
    return router


def train_pages(
    model: base.TwoBlockByteLM,
    router: LearnedCausalHash,
    train: torch.Tensor,
    used_bits: int,
    args: argparse.Namespace,
    seed: int,
) -> base.PageResidualMLP:
    num_pages = 1 if used_bits == 0 else 2**used_bits
    base.set_seed(seed)
    pages = base.PageResidualMLP(
        num_pages,
        d_model=model.d_model,
        hidden_dim=10,
        residual_scale=0.15,
        block_bytes=args.block_bytes,
    )
    for parameter in model.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        pages.parameters(), lr=args.page_lr, weight_decay=1e-4
    )
    model.eval()
    router.eval()
    pages.train()

    for _ in range(args.page_steps):
        x, y = base.get_batch(train, model.context, args.page_batch_size)
        with torch.no_grad():
            hidden = model.first(x)
            page_ids = route(router, hidden, used_bits)
        logits = model.head(model.rest(hidden + pages(page_ids, hidden)))
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return pages


def normalized_entropy(counts: torch.Tensor) -> float:
    raw = counts.double()
    positive = raw[raw > 0]
    if len(raw) == 1:
        return 1.0
    probabilities = positive / positive.sum()
    return float(
        (-(probabilities * probabilities.log()).sum() / math.log(len(raw))).item()
    )


def evaluate_pages(
    model: base.TwoBlockByteLM,
    router: LearnedCausalHash,
    pages: base.PageResidualMLP,
    validation: torch.Tensor,
    used_bits: int,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    # Keep the CE batch RNG isolated from the perturbation-stability diagnostic.
    torch.manual_seed(args.eval_seed)
    values: list[float] = []
    counts = torch.zeros(pages.num_pages)
    model.eval()
    router.eval()
    pages.eval()
    with torch.no_grad():
        for _ in range(args.eval_steps):
            x, y = base.get_batch(
                validation, model.context, args.eval_batch_size
            )
            hidden = model.first(x)
            page_ids = route(router, hidden, used_bits)
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


def routing_stability(
    model: base.TwoBlockByteLM,
    router: LearnedCausalHash,
    validation: torch.Tensor,
    used_bits: int,
    args: argparse.Namespace,
) -> float:
    if used_bits == 0:
        return 1.0

    torch.manual_seed(args.stability_batch_seed)
    hidden_values: list[torch.Tensor] = []
    model.eval()
    router.eval()
    with torch.no_grad():
        for _ in range(args.stability_batches):
            x, _ = base.get_batch(
                validation, model.context, args.stability_batch_size
            )
            hidden_values.append(model.first(x).reshape(-1, model.d_model))
        hidden = torch.cat(hidden_values, dim=0)
        scale = hidden.std(dim=0, unbiased=False).clamp_min(1e-4)
        clean = route(router, hidden, used_bits)
        torch.manual_seed(args.stability_noise_seed)
        noisy = route(
            router,
            hidden
            + args.router_noise_std * scale * torch.randn_like(hidden),
            used_bits,
        )
    return float((clean == noisy).float().mean().item())


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

    router_checkpoint = Path(args.router_checkpoint)
    if args.retrain_router or not router_checkpoint.exists():
        router = train_router(model, train, args)
        router_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(router.state_dict(), router_checkpoint)
    else:
        router = LearnedCausalHash(
            model.d_model, args.router_hidden_dim, args.max_bits
        )
        router.load_state_dict(
            torch.load(router_checkpoint, weights_only=True)
        )
        router.eval()
        for parameter in router.parameters():
            parameter.requires_grad = False

    router_params = sum(parameter.numel() for parameter in router.parameters())
    router_macs = (
        model.d_model * args.router_hidden_dim
        + args.router_hidden_dim * args.max_bits
    )
    page_parameters = (
        10 * model.d_model + 10 + model.d_model * 10 + model.d_model
    )
    page_payload_bytes = 4 * page_parameters
    active_page_macs = 2 * model.d_model * 10
    if page_payload_bytes > args.block_bytes:
        raise ValueError("page payload exceeds physical block")

    used_values = [int(value) for value in args.used_bits.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    stability = {
        used_bits: routing_stability(
            model, router, validation, used_bits, args
        )
        for used_bits in used_values
    }

    print("# ParamProbe G3b: learned causal hash, shared frozen router")
    print(f"corpus_bytes={len(raw)}")
    print(f"backbone_validation_ce={base_ce:.8f}")
    print(f"router_parameters={router_params}")
    print(f"fixed_routing_macs_per_token={router_macs}")
    print(f"page_parameters={page_parameters}")
    print(f"page_payload_bytes={page_payload_bytes}")
    print(f"block_bytes={args.block_bytes}")
    print(f"active_page_macs_per_token={active_page_macs}")
    print("q_inference=1")
    print(f"logical_external_bytes_per_token={args.block_bytes}")
    print("router_supervision=causal_hidden_state_only")
    print()
    print(
        "seed\tused_bits\tpages\tvalidation_ce\tutil_entropy\t"
        "dead_pages\troute_stability"
    )

    results: dict[int, list[float]] = {used_bits: [] for used_bits in used_values}
    per_seed: dict[int, list[float]] = {}
    for seed in seeds:
        per_seed[seed] = []
        for used_bits in used_values:
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
            pages = train_pages(
                model, router, train, used_bits, args, seed
            )
            ce, entropy, dead = evaluate_pages(
                model, router, pages, validation, used_bits, args
            )
            num_pages = 1 if used_bits == 0 else 2**used_bits
            results[used_bits].append(ce)
            per_seed[seed].append(ce)
            print(
                f"{seed}\t{used_bits}\t{num_pages}\t{ce:.8f}\t"
                f"{entropy:.8f}\t{dead:.8f}\t{stability[used_bits]:.8f}"
            )

    print()
    print("pages\tvalidation_ce_mean\tvalidation_ce_sample_std")
    for used_bits in used_values:
        values = np.asarray(results[used_bits], dtype=np.float64)
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        num_pages = 1 if used_bits == 0 else 2**used_bits
        print(f"{num_pages}\t{values.mean():.8f}\t{std:.8f}")

    monotone = {
        seed: bool(np.all(np.diff(np.asarray(values)) < 0.0))
        for seed, values in per_seed.items()
    }
    print()
    print(
        "monotone_by_seed="
        + ",".join(f"{seed}:{str(value).lower()}" for seed, value in monotone.items())
    )
    print(
        f"g3b_passed={str(all(monotone.values()) and len(seeds) >= 3).lower()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-path", type=str, default=None)
    parser.add_argument("--corpus-bytes", type=int, default=1_200_000)
    parser.add_argument(
        "--checkpoint", type=str, default=".cache/paramprobe_g3a_base.pt"
    )
    parser.add_argument("--retrain-backbone", action="store_true")
    parser.add_argument(
        "--router-checkpoint",
        type=str,
        default=".cache/paramprobe_g3b_router.pt",
    )
    parser.add_argument("--retrain-router", action="store_true")
    parser.add_argument("--backbone-seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--pretrain-batch-size", type=int, default=32)
    parser.add_argument("--pretrain-lr", type=float, default=3e-3)
    parser.add_argument("--max-bits", type=int, default=6)
    parser.add_argument("--used-bits", type=str, default="0,2,4,6")
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
    parser.add_argument("--seeds", type=str, default="7,8,9")
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--page-steps", type=int, default=180)
    parser.add_argument("--page-batch-size", type=int, default=24)
    parser.add_argument("--page-lr", type=float, default=4e-3)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--stability-batches", type=int, default=20)
    parser.add_argument("--stability-batch-size", type=int, default=24)
    parser.add_argument("--stability-batch-seed", type=int, default=5678)
    parser.add_argument("--stability-noise-seed", type=int, default=4321)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())
