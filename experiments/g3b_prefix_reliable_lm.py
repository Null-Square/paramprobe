"""G3b refinement: prefix-balanced learned causal routing with reliability ordering.

This experiment follows the negative shared-router reconstruction in
``g3b_learned_causal_hash_lm.py``.  The protocol is fixed before the page-training
replication:

1. train one maximum-width six-factor router from causal block-1 hidden states;
2. apply equal Renyi-2 composite-balance pressure to the 2-, 4-, and 6-factor
   prefixes during router training;
3. freeze the router;
4. on a deterministic training-only hidden-state bank, rank the six factors by
   clean-vs-perturbed hard-bit agreement;
5. use the leading 0/2/4/6 factors of that fixed reliability ordering for the
   1/4/16/64-page sweep.

No language-model labels, next-token counterfactual targets, or semantic page
addresses are used to train or order the router.  Router architecture and MACs
remain fixed across the capacity sweep.  Inference reads exactly one 4096-byte
page per token.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import g3b_learned_causal_hash_lm as prior
from paramprobe.routing_regularizers import (
    paired_collision_estimate,
    renyi2_address_deficit,
)


def train_router(
    model: prior.base.TwoBlockByteLM,
    train: torch.Tensor,
    args: argparse.Namespace,
) -> prior.LearnedCausalHash:
    hidden = prior.collect_hidden_bank(
        model,
        train,
        args.router_hidden_batches,
        args.router_hidden_batch_size,
        args.router_hidden_seed,
    )
    feature_scale = hidden.std(dim=0, unbiased=False).clamp_min(1e-4)

    torch.manual_seed(args.router_seed)
    router = prior.LearnedCausalHash(
        model.d_model, args.router_hidden_dim, args.max_bits
    )
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=args.router_lr,
        weight_decay=args.router_weight_decay,
    )
    balance_prefixes = [
        int(value) for value in args.balance_prefixes.split(",")
    ]
    if not balance_prefixes:
        raise ValueError("balance_prefixes must be non-empty")
    if any(value <= 0 or value > args.max_bits for value in balance_prefixes):
        raise ValueError("balance prefix is outside the active factor range")

    router.train()
    for _ in range(args.router_steps):
        idx = torch.randint(0, len(hidden), (args.router_batch_size,))
        clean = hidden[idx]
        first = clean + (
            args.router_noise_std * feature_scale * torch.randn_like(clean)
        )
        second = clean + (
            args.router_noise_std * feature_scale * torch.randn_like(clean)
        )

        p_first = torch.sigmoid(router(first))
        p_second = torch.sigmoid(router(second))
        pooled = 0.5 * (p_first + p_second)

        consistency = 4.0 * torch.mean((p_first - p_second) ** 2)
        prefix_deficits = []
        for width in balance_prefixes:
            collision = paired_collision_estimate(pooled[:, :width])
            prefix_deficits.append(
                renyi2_address_deficit(collision, width)
            )
        balance = torch.stack(prefix_deficits).mean()
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


def reliability_order(
    model: prior.base.TwoBlockByteLM,
    router: prior.LearnedCausalHash,
    train: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank factors using perturbation stability on training hidden states only."""
    hidden = prior.collect_hidden_bank(
        model,
        train,
        args.order_hidden_batches,
        args.order_hidden_batch_size,
        args.order_hidden_seed,
    )
    feature_scale = hidden.std(dim=0, unbiased=False).clamp_min(1e-4)
    router.eval()
    with torch.no_grad():
        clean = router(hidden) > 0
        torch.manual_seed(args.order_noise_seed)
        noisy = (
            router(
                hidden
                + args.router_noise_std
                * feature_scale
                * torch.randn_like(hidden)
            )
            > 0
        )
        stability = (clean == noisy).float().mean(dim=0)

    order = sorted(
        range(args.max_bits),
        key=lambda factor: (-float(stability[factor]), factor),
    )
    return torch.tensor(order, dtype=torch.long), stability


def bits_to_ids(bits: torch.Tensor) -> torch.Tensor:
    ids = torch.zeros(bits.shape[:-1], dtype=torch.long, device=bits.device)
    for factor in range(bits.shape[-1]):
        ids = (ids << 1) | bits[..., factor].long()
    return ids


def route(
    router: prior.LearnedCausalHash,
    hidden: torch.Tensor,
    used_bits: int,
    order: torch.Tensor,
) -> torch.Tensor:
    # Always execute all six router outputs; only the hard-address projection changes.
    logits = router(hidden)
    if used_bits == 0:
        return torch.zeros(
            hidden.shape[:-1], dtype=torch.long, device=hidden.device
        )
    selected = logits[..., order[:used_bits].to(logits.device)] > 0
    return bits_to_ids(selected)


def train_pages(
    model: prior.base.TwoBlockByteLM,
    router: prior.LearnedCausalHash,
    train: torch.Tensor,
    used_bits: int,
    order: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> prior.base.PageResidualMLP:
    num_pages = 1 if used_bits == 0 else 2**used_bits
    prior.base.set_seed(seed)
    pages = prior.base.PageResidualMLP(
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
        x, y = prior.base.get_batch(
            train, model.context, args.page_batch_size
        )
        with torch.no_grad():
            hidden = model.first(x)
            page_ids = route(router, hidden, used_bits, order)
        logits = model.head(model.rest(hidden + pages(page_ids, hidden)))
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return pages


def evaluate_pages(
    model: prior.base.TwoBlockByteLM,
    router: prior.LearnedCausalHash,
    pages: prior.base.PageResidualMLP,
    validation: torch.Tensor,
    used_bits: int,
    order: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    torch.manual_seed(args.eval_seed)
    values: list[float] = []
    counts = torch.zeros(pages.num_pages)
    model.eval()
    router.eval()
    pages.eval()
    with torch.no_grad():
        for _ in range(args.eval_steps):
            x, y = prior.base.get_batch(
                validation, model.context, args.eval_batch_size
            )
            hidden = model.first(x)
            page_ids = route(router, hidden, used_bits, order)
            logits = model.head(model.rest(hidden + pages(page_ids, hidden)))
            values.append(
                F.cross_entropy(
                    logits.reshape(-1, 256), y.reshape(-1)
                ).item()
            )
            counts += torch.bincount(
                page_ids.reshape(-1).cpu(), minlength=pages.num_pages
            )
    return (
        float(np.mean(values)),
        prior.normalized_entropy(counts),
        float((counts == 0).double().mean().item()),
    )


def routing_stability(
    model: prior.base.TwoBlockByteLM,
    router: prior.LearnedCausalHash,
    validation: torch.Tensor,
    used_bits: int,
    order: torch.Tensor,
    args: argparse.Namespace,
) -> float:
    if used_bits == 0:
        return 1.0
    torch.manual_seed(args.stability_batch_seed)
    values: list[torch.Tensor] = []
    model.eval()
    router.eval()
    with torch.no_grad():
        for _ in range(args.stability_batches):
            x, _ = prior.base.get_batch(
                validation, model.context, args.stability_batch_size
            )
            values.append(model.first(x).reshape(-1, model.d_model))
        hidden = torch.cat(values, dim=0)
        scale = hidden.std(dim=0, unbiased=False).clamp_min(1e-4)
        clean = route(router, hidden, used_bits, order)
        torch.manual_seed(args.stability_noise_seed)
        noisy = route(
            router,
            hidden
            + args.router_noise_std * scale * torch.randn_like(hidden),
            used_bits,
            order,
        )
    return float((clean == noisy).float().mean().item())


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    prior.base.set_seed(args.backbone_seed)
    raw = prior.base.load_corpus(args.text_path, args.corpus_bytes)
    data = torch.tensor(list(raw), dtype=torch.long)
    split = int(0.9 * len(data))
    train, validation = data[:split], data[split:]

    model = prior.base.TwoBlockByteLM(
        d_model=args.d_model, heads=args.heads, context=args.context
    )
    checkpoint = Path(args.checkpoint)
    if args.retrain_backbone or not checkpoint.exists():
        prior.base.pretrain_backbone(
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

    base_ce = prior.base.evaluate_backbone(
        model, validation, args.eval_steps, args.eval_batch_size
    )

    router_checkpoint = Path(args.router_checkpoint)
    if args.retrain_router or not router_checkpoint.exists():
        router = train_router(model, train, args)
        router_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(router.state_dict(), router_checkpoint)
    else:
        router = prior.LearnedCausalHash(
            model.d_model, args.router_hidden_dim, args.max_bits
        )
        router.load_state_dict(
            torch.load(router_checkpoint, weights_only=True)
        )
        router.eval()
        for parameter in router.parameters():
            parameter.requires_grad = False

    order, factor_stability = reliability_order(model, router, train, args)
    used_values = [int(value) for value in args.used_bits.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    stability = {
        used_bits: routing_stability(
            model, router, validation, used_bits, order, args
        )
        for used_bits in used_values
    }

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

    print("# ParamProbe G3b: prefix balance + reliability ordering")
    print(f"corpus_bytes={len(raw)}")
    print(f"backbone_validation_ce={base_ce:.8f}")
    print(f"router_parameters={router_params}")
    print(f"fixed_routing_macs_per_token={router_macs}")
    print(f"balance_prefixes={args.balance_prefixes}")
    print(
        "factor_stability="
        + ",".join(f"{value:.8f}" for value in factor_stability.tolist())
    )
    print("reliability_order=" + ",".join(map(str, order.tolist())))
    print(f"page_parameters={page_parameters}")
    print(f"page_payload_bytes={page_payload_bytes}")
    print(f"block_bytes={args.block_bytes}")
    print(f"active_page_macs_per_token={active_page_macs}")
    print("q_inference=1")
    print(f"logical_external_bytes_per_token={args.block_bytes}")
    print("router_supervision=causal_hidden_state_only")
    print("factor_ordering=training_hidden_perturbation_stability_only")
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
                model, router, train, used_bits, order, args, seed
            )
            ce, entropy, dead = evaluate_pages(
                model, router, pages, validation, used_bits, order, args
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
        + ",".join(
            f"{seed}:{str(value).lower()}" for seed, value in monotone.items()
        )
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
        default=".cache/paramprobe_g3b_prefix_reliable_router.pt",
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
    parser.add_argument("--balance-prefixes", type=str, default="2,4,6")
    parser.add_argument("--order-hidden-batches", type=int, default=40)
    parser.add_argument("--order-hidden-batch-size", type=int, default=24)
    parser.add_argument("--order-hidden-seed", type=int, default=97531)
    parser.add_argument("--order-noise-seed", type=int, default=86420)
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
