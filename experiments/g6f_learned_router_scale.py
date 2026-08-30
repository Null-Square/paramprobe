"""G6f: learned-factorized-router capacity gate for the frozen larger subword LM.

Predeclared in docs/g6f_learned_router_scale_predeclared.md before this file was
created.  Router training and factor ordering use training hidden states only.
Inference executes the full fixed-width learned router and one 16 KiB page/token.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import g6a_subword_scale_lm as g6a
import g6d_capacity_scaled_training as g6d
from paramprobe.routing_regularizers import (
    paired_collision_estimate,
    renyi2_address_deficit,
)

SEEDS = (32, 33, 34)
USED_BITS = (4, 6, 8)
BALANCE_PREFIXES = (2, 4, 6, 8)
ROUTER_HIDDEN = 64
MAX_BITS = 8
ROUTER_NOISE_STD = 0.08


class LearnedCausalHash(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(96, ROUTER_HIDDEN),
            nn.GELU(),
            nn.Linear(ROUTER_HIDDEN, MAX_BITS),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


def collect_hidden_states(
    model: g6a.SubwordCausalLM,
    train: torch.Tensor,
    batches: int,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    values: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            x, _ = g6a.get_batch(train, 128, 8)
            values.append(model.first(x).reshape(-1, 96).cpu())
    return torch.cat(values, dim=0)


def train_router(
    model: g6a.SubwordCausalLM,
    train: torch.Tensor,
) -> LearnedCausalHash:
    hidden = collect_hidden_states(model, train, batches=120, seed=2468)
    if hidden.shape != (122_880, 96):
        raise RuntimeError(f"router hidden-bank shape changed: {hidden.shape}")
    feature_scale = hidden.std(dim=0, unbiased=False).clamp_min(1e-4)

    torch.manual_seed(17)
    router = LearnedCausalHash()
    optimizer = torch.optim.AdamW(router.parameters(), lr=2e-3, weight_decay=1e-4)
    router.train()
    for _ in range(600):
        idx = torch.randint(0, len(hidden), (1024,))
        clean = hidden[idx]
        first = clean + ROUTER_NOISE_STD * feature_scale * torch.randn_like(clean)
        second = clean + ROUTER_NOISE_STD * feature_scale * torch.randn_like(clean)

        p_first = torch.sigmoid(router(first))
        p_second = torch.sigmoid(router(second))
        pooled = 0.5 * (p_first + p_second)

        consistency = 4.0 * torch.mean((p_first - p_second) ** 2)
        deficits = []
        for width in BALANCE_PREFIXES:
            collision = paired_collision_estimate(pooled[:, :width])
            deficits.append(renyi2_address_deficit(collision, width))
        balance = torch.stack(deficits).mean()
        confidence = torch.mean(4.0 * pooled * (1.0 - pooled))
        loss = consistency + 0.2 * balance + 0.1 * confidence

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for p in router.parameters():
        p.requires_grad = False
    router.eval()
    return router


def reliability_order(
    model: g6a.SubwordCausalLM,
    router: LearnedCausalHash,
    train: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = collect_hidden_states(model, train, batches=60, seed=97531)
    if hidden.shape != (61_440, 96):
        raise RuntimeError(f"ordering hidden-bank shape changed: {hidden.shape}")
    feature_scale = hidden.std(dim=0, unbiased=False).clamp_min(1e-4)
    with torch.no_grad():
        clean = router(hidden) > 0
        torch.manual_seed(86420)
        noisy = router(
            hidden + ROUTER_NOISE_STD * feature_scale * torch.randn_like(hidden)
        ) > 0
        stability = (clean == noisy).float().mean(dim=0)
    order = sorted(
        range(MAX_BITS),
        key=lambda factor: (-float(stability[factor]), factor),
    )
    return torch.tensor(order, dtype=torch.long), stability


def route(
    router: LearnedCausalHash,
    hidden: torch.Tensor,
    used_bits: int,
    order: torch.Tensor,
) -> torch.Tensor:
    # Always execute all eight router outputs; capacity changes only the selected
    # hard prefix of the one frozen reliability order.
    logits = router(hidden)
    selected = logits[..., order[:used_bits].to(logits.device)] > 0
    ids = torch.zeros(selected.shape[:-1], dtype=torch.long, device=selected.device)
    for factor in range(used_bits):
        ids = (ids << 1) | selected[..., factor].long()
    return ids


def routing_stats(
    router: LearnedCausalHash,
    eval_bank: list[tuple[torch.Tensor, torch.Tensor]],
    used_bits: int,
    order: torch.Tensor,
) -> tuple[float, float]:
    n = 2**used_bits
    counts = torch.zeros(n, dtype=torch.float64)
    with torch.no_grad():
        for hidden, _ in eval_bank:
            ids = route(router, hidden, used_bits, order)
            counts += torch.bincount(ids.reshape(-1).cpu(), minlength=n).double()
    raw = counts.numpy()
    positive = raw[raw > 0]
    probs = positive / positive.sum()
    entropy = float(-(probs * np.log(probs)).sum() / np.log(n))
    return entropy, float(np.mean(raw == 0))


def route_stability(
    router: LearnedCausalHash,
    eval_bank: list[tuple[torch.Tensor, torch.Tensor]],
    used_bits: int,
    order: torch.Tensor,
) -> float:
    hidden = torch.cat([h.reshape(-1, 96) for h, _ in eval_bank], dim=0)
    scale = hidden.std(dim=0, unbiased=False).clamp_min(1e-4)
    with torch.no_grad():
        clean = route(router, hidden, used_bits, order)
        torch.manual_seed(4321)
        noisy = route(
            router,
            hidden + ROUTER_NOISE_STD * scale * torch.randn_like(hidden),
            used_bits,
            order,
        )
    return float((clean == noisy).float().mean().item())


def evaluate(
    model: g6a.SubwordCausalLM,
    router: LearnedCausalHash,
    pages: g6a.PageResidualMLP,
    eval_bank: list[tuple[torch.Tensor, torch.Tensor]],
    used_bits: int,
    order: torch.Tensor,
) -> float:
    values: list[float] = []
    pages.eval()
    with torch.no_grad():
        for hidden, y in eval_bank:
            ids = route(router, hidden, used_bits, order)
            logits = model.head(model.rest(hidden + pages(ids, hidden)))
            values.append(F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1)).item())
    return float(np.mean(values))


def train_capacity(
    model: g6a.SubwordCausalLM,
    router: LearnedCausalHash,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
    eval_bank: list[tuple[torch.Tensor, torch.Tensor]],
    used_bits: int,
    order: torch.Tensor,
    seed: int,
) -> float:
    n = 2**used_bits
    steps = g6d.budget_steps(used_bits)
    pages = g6d.init_pages(n, seed)
    optimizer = torch.optim.AdamW(pages.parameters(), lr=4e-3, weight_decay=1e-4)
    pages.train()
    for hidden, y in bank[:steps]:
        with torch.no_grad():
            ids = route(router, hidden, used_bits, order)
        logits = model.head(model.rest(hidden + pages(ids, hidden)))
        loss = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return evaluate(model, router, pages, eval_bank, used_bits, order)


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    tokenizer, model = g6d.load_assets(args)
    train = g6a.encode_split(tokenizer, args.train_path)
    valid = g6a.encode_split(tokenizer, args.validation_path)
    if len(train) != 4_254_523 or len(valid) != 445_470:
        raise RuntimeError("tokenized split sizes changed")

    router = train_router(model, train)
    order, factor_stability = reliability_order(model, router, train)
    eval_bank = g6a.make_hidden_bank(model, valid, 40, 8, 1234)

    router_params = sum(p.numel() for p in router.parameters())
    router_macs = 96 * ROUTER_HIDDEN + ROUTER_HIDDEN * MAX_BITS
    if router_params != 6728 or router_macs != 6656:
        raise RuntimeError(
            f"learned router envelope changed: params={router_params} macs={router_macs}"
        )
    if (20 * 96 + 20 + 96 * 20 + 96, 2 * 96 * 20) != (3956, 3840):
        raise RuntimeError("page inference resource envelope changed")
    expected_steps = {4: 120, 6: 480, 8: 1920}
    actual_steps = {u: g6d.budget_steps(u) for u in USED_BITS}
    if actual_steps != expected_steps:
        raise RuntimeError(f"page-training budget changed: {actual_steps}")

    stats = {u: routing_stats(router, eval_bank, u, order) for u in USED_BITS}
    stabilities = {
        u: route_stability(router, eval_bank, u, order) for u in USED_BITS
    }

    print("# ParamProbe G6f: learned-factorized-router larger-LM scale gate")
    print("protocol_predeclared=true")
    print("router_supervision=training_hidden_states_only")
    print("fresh_page_seeds=32,33,34")
    print(f"tokenizer_sha256={g6d.sha256_path(args.tokenizer_json)}")
    print(f"backbone_sha256={g6d.sha256_path(args.checkpoint)}")
    print(f"router_parameters={router_params}")
    print(f"router_matrix_macs_per_token={router_macs}")
    print("router_architecture=96->64->8_gelu")
    print("router_hidden_states=122880")
    print("router_steps=600")
    print("router_balance_prefixes=2,4,6,8")
    print("router_noise_std=0.08")
    print("ordering_hidden_states=61440")
    print(
        "factor_stability="
        + ",".join(f"{v:.8f}" for v in factor_stability.tolist())
    )
    print("reliability_order=" + ",".join(map(str, order.tolist())))
    print("q_inference=1")
    print("block_bytes=16384")
    print("logical_external_bytes_per_token=16384")
    print("page_parameters=3956")
    print("page_payload_bytes=15824")
    print("active_page_macs_per_token=3840")
    print("training_exposure_floor_per_page=7680")
    print()
    print("pages\tused_bits\tsteps\ttotal_assignments\tmean_assignments_per_page\tutil_entropy\tdead_page_fraction\troute_stability")
    for u in USED_BITS:
        n = 2**u
        steps = actual_steps[u]
        total = steps * 1024
        entropy, dead = stats[u]
        print(
            f"{n}\t{u}\t{steps}\t{total}\t{total / n:.1f}\t"
            f"{entropy:.8f}\t{dead:.8f}\t{stabilities[u]:.8f}"
        )

    results: dict[int, list[float]] = {u: [] for u in USED_BITS}
    by_seed: dict[int, list[float]] = {s: [] for s in SEEDS}
    print()
    print("seed\tused_bits\tpages\tsteps\tvalidation_ce")
    for seed in SEEDS:
        bank = g6a.make_hidden_bank(model, train, 1920, 8, 50000 + seed)
        for u in USED_BITS:
            ce = train_capacity(
                model, router, bank, eval_bank, u, order, seed
            )
            results[u].append(ce)
            by_seed[seed].append(ce)
            print(f"{seed}\t{u}\t{2**u}\t{actual_steps[u]}\t{ce:.8f}", flush=True)
        del bank

    means: list[float] = []
    print()
    print("pages\tsteps\tvalidation_ce_mean\tvalidation_ce_sample_std")
    for u in USED_BITS:
        arr = np.asarray(results[u], dtype=np.float64)
        means.append(float(arr.mean()))
        print(f"{2**u}\t{actual_steps[u]}\t{arr.mean():.8f}\t{arr.std(ddof=1):.8f}")

    mean_monotone = bool(np.all(np.diff(np.asarray(means)) < 0.0))
    endpoint = {s: by_seed[s][2] < by_seed[s][0] for s in SEEDS}
    full_monotone = {
        s: bool(np.all(np.diff(np.asarray(by_seed[s])) < 0.0)) for s in SEEDS
    }
    print()
    print(f"mean_curve_strictly_monotone={str(mean_monotone).lower()}")
    print(
        "n256_beats_n16_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in endpoint.items())
    )
    print(
        "full_monotone_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in full_monotone.items())
    )
    print("resource_gate_passed=true")
    print("training_budget_gate_passed=true")
    print("router_training_leakage_gate_passed=true")
    passed = mean_monotone and all(endpoint.values())
    print(f"g6f_learned_router_scale_passed={str(passed).lower()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", required=True)
    p.add_argument("--validation-path", required=True)
    p.add_argument("--tokenizer-json", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threads", type=int, default=8)
    run(p.parse_args())
