"""G2f: fixed-budget counterfactual utility training, hard one-page inference.

The router receives no semantic address labels. During training we evaluate four
candidate parameter pages per example (for every multi-page condition), measure
the task loss each candidate would produce, and distill that counterfactual
utility into the factorized router. At inference the router selects exactly one
4 KiB page.

Candidate budget for N >= 4 is fixed at four pages:
  1. the current hard address;
  2-4. one-bit neighboring addresses, prioritizing the least-confident factors;
  4. when fewer than three distinct one-bit neighbors exist, an exploratory
     random route fills the remaining slot.

The task-utility target is a softmin over per-candidate task losses. Page
parameters are trained using that detached utility distribution, and the router
is trained to match it using the factorized log-probabilities of the same
candidate routes. Composite-address Renyi-2 balancing is retained to prevent
page collapse.

This is a training algorithm, not an inference-budget relaxation: q_infer=1 and
logical external parameter traffic remains exactly one block per invocation.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

import g2d_task_only_top2 as base
from paramprobe.routing_regularizers import (
    paired_collision_estimate,
    renyi2_address_deficit,
)


def candidate_pages_and_scores(
    logits: torch.Tensor, used_bits: int, candidate_budget: int
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = logits.shape[0]
    if used_bits == 0:
        return (
            torch.zeros(batch, 1, dtype=torch.long, device=logits.device),
            torch.zeros(batch, 1, dtype=logits.dtype, device=logits.device),
        )
    if candidate_budget < 2:
        raise ValueError("candidate_budget must be >= 2 for multi-page routing")

    hard = logits[:, :used_bits] > 0
    candidates = [hard]

    # Local counterfactuals: flip the least-confident address factors first.
    factor_order = logits[:, :used_bits].abs().argsort(dim=1)
    for rank in range(min(candidate_budget - 1, used_bits)):
        neighbor = hard.clone()
        factor = factor_order[:, rank]
        neighbor[torch.arange(batch, device=logits.device), factor] = ~neighbor[
            torch.arange(batch, device=logits.device), factor
        ]
        candidates.append(neighbor)

    # For very small address spaces there may be fewer local neighbors than the
    # fixed training budget. Fill the remaining slots with exploratory routes.
    while len(candidates) < candidate_budget:
        candidates.append(torch.rand_like(hard, dtype=torch.float32) > 0.5)

    bits = torch.stack(candidates[:candidate_budget], dim=1)
    ids = torch.zeros(
        (batch, candidate_budget), dtype=torch.long, device=logits.device
    )
    for factor in range(used_bits):
        ids = (ids << 1) | bits[:, :, factor].long()

    active_logits = logits[:, None, :used_bits].expand(
        -1, candidate_budget, -1
    )
    signed_logits = torch.where(bits, active_logits, -active_logits)
    scores = -torch.nn.functional.softplus(-signed_logits).sum(dim=-1)
    return ids, scores


def run_one(args: argparse.Namespace, seed: int, used_bits: int) -> dict[str, float]:
    base.set_seed(seed)
    rng = np.random.default_rng(seed)

    if args.num_items != 2**args.max_bits:
        raise ValueError("G2f currently uses num_items == 2^max_bits")

    # Adversarial control: semantic prototypes and teacher functions are sampled
    # independently. No hand-crafted relation tells the router which page should
    # own which item; route utility must be learned from task performance.
    base_context = rng.normal(size=(args.num_items, args.context_dim)).astype(np.float32)
    base_context /= np.linalg.norm(base_context, axis=1, keepdims=True) + 1e-12
    base_context += rng.normal(
        scale=args.item_nuisance, size=base_context.shape
    ).astype(np.float32)

    teacher = base.make_teacher(
        args.num_items,
        args.query_dim,
        args.page_hidden_dim,
        args.output_dim,
        rng,
    )

    train_items_np = np.repeat(np.arange(args.num_items), args.train_queries_per_item)
    val_items_np = np.repeat(np.arange(args.num_items), args.val_queries_per_item)
    train_items = torch.from_numpy(train_items_np).long()
    val_items = torch.from_numpy(val_items_np).long()

    train_context = np.repeat(base_context, args.train_queries_per_item, axis=0)
    train_context += rng.normal(
        scale=args.context_noise, size=train_context.shape
    ).astype(np.float32)
    val_context = np.repeat(base_context, args.val_queries_per_item, axis=0)
    val_context += rng.normal(
        scale=args.context_noise, size=val_context.shape
    ).astype(np.float32)

    x_train = torch.from_numpy(
        rng.normal(size=(len(train_items_np), args.query_dim)).astype(np.float32)
    )
    x_val = torch.from_numpy(
        rng.normal(size=(len(val_items_np), args.query_dim)).astype(np.float32)
    )
    with torch.no_grad():
        y_train = base.teacher_forward(teacher, train_items, x_train)
        y_val = base.teacher_forward(teacher, val_items, x_val)

    context_train = torch.from_numpy(train_context)
    context_val = torch.from_numpy(val_context)
    num_pages = 1 if used_bits == 0 else 1 << used_bits

    router = base.FixedRouter(args.context_dim, args.router_hidden_dim, args.max_bits)
    pages = base.PageMLPTable(
        num_pages,
        args.query_dim,
        args.page_hidden_dim,
        args.output_dim,
    )
    page_payload_bytes = pages.page_parameters * 4
    if page_payload_bytes > args.block_bytes:
        raise ValueError("page MLP does not fit inside block_bytes")

    optimizer = torch.optim.Adam(
        [
            {"params": router.parameters(), "lr": args.router_lr},
            {"params": pages.parameters(), "lr": args.page_lr},
        ]
    )

    for _ in range(args.epochs):
        order = torch.randperm(len(train_items))
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            logits = router(context_train[idx])
            candidate_ids, candidate_scores = candidate_pages_and_scores(
                logits, used_bits, args.candidate_budget
            )

            outputs = []
            for candidate in range(candidate_ids.shape[1]):
                outputs.append(pages(candidate_ids[:, candidate], x_train[idx]))
            stacked_outputs = torch.stack(outputs, dim=1)
            candidate_losses = torch.mean(
                (stacked_outputs - y_train[idx, None, :]) ** 2, dim=-1
            )

            if candidate_ids.shape[1] == 1:
                task_loss = candidate_losses.mean()
                router_loss = task_loss * 0.0
            else:
                utility = torch.softmax(
                    -candidate_losses.detach() / args.utility_temperature,
                    dim=1,
                )
                task_loss = torch.mean(
                    torch.sum(utility * candidate_losses, dim=1)
                )
                router_log_probs = torch.log_softmax(
                    candidate_scores / args.router_temperature,
                    dim=1,
                )
                router_loss = -torch.mean(
                    torch.sum(utility * router_log_probs, dim=1)
                )

            if used_bits == 0:
                balance = task_loss * 0.0
            else:
                probabilities = torch.sigmoid(logits[:, :used_bits])
                collision = paired_collision_estimate(probabilities)
                balance = renyi2_address_deficit(collision, used_bits)

            loss = (
                task_loss
                + args.router_utility_weight * router_loss
                + args.balance_weight * balance
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    router.eval()
    pages.eval()
    with torch.no_grad():
        logits = router(context_val)
        selected_pages = base.bits_to_page(logits > 0, used_bits)
        prediction = pages(selected_pages, x_val)
        normalized_mse = float(
            torch.mean((prediction - y_val) ** 2) / torch.mean(y_val**2)
        )
        counts = torch.bincount(selected_pages, minlength=num_pages).cpu().numpy()

    routed = selected_pages.cpu().numpy().reshape(
        args.num_items, args.val_queries_per_item
    )
    modal_pages: list[int] = []
    stability: list[float] = []
    for row in routed:
        local = np.bincount(row, minlength=num_pages)
        modal_pages.append(int(local.argmax()))
        stability.append(float(local.max() / len(row)))

    return {
        "seed": float(seed),
        "used_bits": float(used_bits),
        "pages": float(num_pages),
        "normalized_mse": normalized_mse,
        "utilization_entropy": base.normalized_entropy(counts),
        "dead_page_fraction": float(np.mean(counts == 0)),
        "routing_stability": float(np.mean(stability)),
        "modal_item_collision_fraction": float(
            1.0 - len(set(modal_pages)) / args.num_items
        ),
        "page_payload_bytes": float(page_payload_bytes),
    }


def run(args: argparse.Namespace) -> None:
    controller_params = (
        args.context_dim * args.router_hidden_dim
        + args.router_hidden_dim
        + args.router_hidden_dim * args.max_bits
        + args.max_bits
    )
    routing_macs = (
        args.context_dim * args.router_hidden_dim
        + args.router_hidden_dim * args.max_bits
    )
    operator_macs = (
        args.query_dim * args.page_hidden_dim
        + args.page_hidden_dim * args.output_dim
    )

    print("# ParamProbe G2f: fixed-budget counterfactual utility routing")
    print(f"controller_params={controller_params}")
    print(f"routing_macs_per_query={routing_macs}")
    print(f"active_operator_macs_per_inference={operator_macs}")
    print(f"training_candidate_budget={args.candidate_budget}")
    print("q_inference=1")
    print(f"block_bytes={args.block_bytes}")
    print(f"logical_bytes_per_inference={args.block_bytes}")
    print("router_supervision=task_counterfactuals_only")
    print("address_labels=none")
    print()
    print(
        "seed\tused_bits\tpages\texternal_MiB\tnmse\tutil_entropy\t"
        "dead_pages\troute_stability\tmodal_item_collision"
    )

    seeds = [int(value) for value in args.seeds.split(",")]
    sweep = [0] + list(range(args.min_bits, args.max_bits + 1, args.bit_step))
    if sweep[-1] != args.max_bits:
        sweep.append(args.max_bits)

    for seed in seeds:
        for used_bits in sweep:
            result = run_one(args, seed, used_bits)
            external_mib = result["pages"] * args.block_bytes / (1024**2)
            print(
                f"{seed}\t{used_bits}\t{int(result['pages'])}\t{external_mib:.6f}\t"
                f"{result['normalized_mse']:.8f}\t{result['utilization_entropy']:.8f}\t"
                f"{result['dead_page_fraction']:.8f}\t{result['routing_stability']:.8f}\t"
                f"{result['modal_item_collision_fraction']:.8f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-items", type=int, default=64)
    parser.add_argument("--max-bits", type=int, default=6)
    parser.add_argument("--min-bits", type=int, default=2)
    parser.add_argument("--bit-step", type=int, default=2)
    parser.add_argument("--context-dim", type=int, default=32)
    parser.add_argument("--router-hidden-dim", type=int, default=64)
    parser.add_argument("--query-dim", type=int, default=8)
    parser.add_argument("--page-hidden-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--train-queries-per-item", type=int, default=64)
    parser.add_argument("--val-queries-per-item", type=int, default=24)
    parser.add_argument("--context-noise", type=float, default=0.08)
    parser.add_argument("--item-nuisance", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--router-lr", type=float, default=3e-3)
    parser.add_argument("--page-lr", type=float, default=1e-2)
    parser.add_argument("--candidate-budget", type=int, default=4)
    parser.add_argument("--utility-temperature", type=float, default=0.05)
    parser.add_argument("--router-temperature", type=float, default=0.5)
    parser.add_argument("--router-utility-weight", type=float, default=1.0)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--seeds", type=str, default="1,2,3")
    run(parser.parse_args())
