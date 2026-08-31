"""G7b: balanced q=1 product-key challenge.

Predeclared in docs/g7b_balanced_product_key_predeclared.md and its numerical
addendum before this implementation.  G7b keeps the G7 d_key=6 inference
router/page envelope fixed and changes only training: the product-key router is
trained label-free to form stable, balanced, confident 256-way product
partitions, then frozen before page training.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import g6a_subword_scale_lm as g6a
import g7_hard_budget_product_key as g7

SEEDS = (35, 36, 37)
ROUTER_STEPS = 600
ROUTER_BATCH = 1024
ROUTER_LR = 2e-3
ROUTER_WD = 1e-4
NOISE_STD = 0.08
CONSISTENCY_WEIGHT = 1.0
BALANCE_WEIGHT = 0.2
CONFIDENCE_WEIGHT = 0.1
EPS = 1e-12
LOG_N = math.log(256.0)
UTIL_ENTROPY_MIN = 0.85
DEAD_MAX = 0.05


def soft_product_distribution(
    router: g7.ProductKeyRouter, hidden: torch.Tensor
) -> torch.Tensor:
    flat = hidden.reshape(-1, 96)
    q = router.query_bn(router.query(flat))
    q1, q2 = q[:, :3], q[:, 3:]
    scores1 = q1 @ router.keys1.t()
    scores2 = q2 @ router.keys2.t()
    p1 = torch.softmax(scores1, dim=-1)
    p2 = torch.softmax(scores2, dim=-1)
    return (p1.unsqueeze(2) * p2.unsqueeze(1)).reshape(-1, 256)


def entropy_rows(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p + EPS)).sum(dim=-1)


def train_balanced_router(
    model: g6a.SubwordCausalLM,
    train: torch.Tensor,
    seed: int,
) -> tuple[g7.ProductKeyRouter, tuple[float, float, float, float]]:
    bank = g6a.make_hidden_bank(model, train, 80, 8, 2468)
    hidden = torch.cat([h.reshape(-1, 96) for h, _ in bank], dim=0)
    if hidden.shape != (81_920, 96):
        raise RuntimeError(f"router hidden bank changed: {tuple(hidden.shape)}")
    feature_scale = hidden.std(dim=0, unbiased=False).clamp_min(1e-4)

    router = g7.ProductKeyRouter(d_key=6, seed=seed)
    if router.learned_parameter_count != 678 or router.matrix_macs_per_token != 672:
        raise RuntimeError("G7b product-key inference resource envelope changed")
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=ROUTER_LR, weight_decay=ROUTER_WD
    )

    torch.manual_seed(71000 + seed)
    router.train()
    final = (float("nan"),) * 4
    for _ in range(ROUTER_STEPS):
        idx = torch.randint(0, hidden.shape[0], (ROUTER_BATCH,))
        clean = hidden[idx]
        a = clean + NOISE_STD * feature_scale * torch.randn_like(clean)
        b = clean + NOISE_STD * feature_scale * torch.randn_like(clean)
        p_a = soft_product_distribution(router, a)
        p_b = soft_product_distribution(router, b)
        p_bar = 0.5 * (p_a + p_b)

        consistency = 256.0 * torch.mean((p_a - p_b) ** 2)
        marginal = p_bar.mean(dim=0)
        balance = LOG_N - entropy_rows(marginal.unsqueeze(0)).squeeze(0)
        confidence = entropy_rows(p_bar).mean() / LOG_N
        loss = (
            CONSISTENCY_WEIGHT * consistency
            + BALANCE_WEIGHT * balance
            + CONFIDENCE_WEIGHT * confidence
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final = (
            float(loss.detach()),
            float(consistency.detach()),
            float(balance.detach()),
            float(confidence.detach()),
        )

    for p in router.parameters():
        p.requires_grad = False
    router.eval()
    return router, final


def train_pages_frozen_pk(
    model: g6a.SubwordCausalLM,
    router: g7.ProductKeyRouter,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
    seed: int,
) -> g6a.PageResidualMLP:
    pages = g7.init_pages(seed)
    optimizer = torch.optim.AdamW(
        pages.parameters(), lr=g7.PAGE_LR, weight_decay=g7.WEIGHT_DECAY
    )
    pages.train()
    router.eval()
    for hidden, y in bank:
        with torch.no_grad():
            ids, gate = router(hidden)
        residual = pages(ids, hidden) * gate.unsqueeze(-1)
        logits = model.head(model.rest(hidden + residual))
        loss = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return pages


def hard_stats_on_hidden(
    router: g7.ProductKeyRouter, hidden: torch.Tensor
) -> tuple[float, float]:
    router.eval()
    with torch.no_grad():
        ids, _ = router(hidden)
    return g7.utilization([ids])


def route_stability_pair(
    fixed_projection: torch.Tensor,
    fixed_thresholds: torch.Tensor,
    router: g7.ProductKeyRouter,
    eval_bank: list[tuple[torch.Tensor, torch.Tensor]],
    seed: int,
) -> tuple[float, float]:
    hidden = torch.cat([h.reshape(-1, 96) for h, _ in eval_bank], dim=0)
    scale = hidden.std(dim=0, unbiased=False).clamp_min(1e-4)
    torch.manual_seed(72000 + seed)
    noise = NOISE_STD * scale * torch.randn_like(hidden)
    with torch.no_grad():
        fixed_clean = g6a.route(hidden, fixed_projection, fixed_thresholds, 8)
        fixed_noisy = g6a.route(
            hidden + noise, fixed_projection, fixed_thresholds, 8
        )
        pk_clean, _ = router(hidden)
        pk_noisy, _ = router(hidden + noise)
    return (
        float((fixed_clean == fixed_noisy).float().mean().item()),
        float((pk_clean == pk_noisy).float().mean().item()),
    )


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    tokenizer, model = g7.load_assets(args)
    train = g6a.encode_split(tokenizer, args.train_path)
    valid = g6a.encode_split(tokenizer, args.validation_path)
    if len(train) != 4_254_523 or len(valid) != 445_470:
        raise RuntimeError("tokenized split sizes changed")

    projection, thresholds = g6a.build_balanced_hash(
        model, train, max_bits=8, calibration_batches=40, batch_size=8
    )
    eval_bank = g6a.make_hidden_bank(model, valid, 40, 8, 1234)

    if projection.numel() + thresholds.numel() != 776:
        raise RuntimeError("fixed router metadata changed")
    prototype = g7.init_pages(SEEDS[0])
    if prototype.page_parameters != 3956 or prototype.page_payload_bytes != 15824:
        raise RuntimeError("page resource envelope changed")

    print("# ParamProbe G7b: balanced q=1 product-key challenge")
    print("protocol_predeclared=true")
    print("fresh_seeds=35,36,37")
    print(f"tokenizer_sha256={g7.sha256_path(args.tokenizer_json)}")
    print(f"backbone_sha256={g7.sha256_path(args.checkpoint)}")
    print("pages=256")
    print("q_inference=1")
    print("block_bytes=16384")
    print("logical_external_bytes_per_token=16384")
    print("page_parameters=3956")
    print("page_payload_bytes=15824")
    print("active_page_macs_per_token=3840")
    print("fixed_router_metadata_bytes=3104")
    print("fixed_router_macs_per_token=768")
    print("pk_router_parameter_bytes=2712")
    print("pk_batchnorm_buffer_bytes=56")
    print("pk_router_macs_per_token=672")
    print("router_pretrain_steps=600")
    print("router_pretrain_batch_size=1024")
    print("router_pretrain_hidden_states=81920")
    print("router_pretrain_supervision=training_hidden_states_only")
    print("router_pretrain_consistency_weight=1.0")
    print("router_pretrain_balance_weight=0.2")
    print("router_pretrain_confidence_weight=0.1")
    print("router_pretrain_noise_std=0.08")
    print("page_training_steps=1920")
    print("mean_routed_page_assignments_per_page=7680")
    print("utilization_entropy_gate_min=0.85")
    print("dead_page_fraction_gate_max=0.05")
    print()
    print(
        "seed\trouter_loss\tconsistency\tbalance\tconfidence\t"
        "train_route_entropy\ttrain_dead_fraction"
    )

    routers: dict[int, g7.ProductKeyRouter] = {}
    for seed in SEEDS:
        router, final = train_balanced_router(model, train, seed)
        # Training-only diagnostic on the same bank; not used for model selection.
        diag_bank = g6a.make_hidden_bank(model, train, 80, 8, 2468)
        diag_hidden = torch.cat([h.reshape(-1, 96) for h, _ in diag_bank], dim=0)
        ent, dead = hard_stats_on_hidden(router, diag_hidden)
        routers[seed] = router
        print(
            f"{seed}\t{final[0]:.8f}\t{final[1]:.8f}\t{final[2]:.8f}\t"
            f"{final[3]:.8f}\t{ent:.8f}\t{dead:.8f}",
            flush=True,
        )

    print()
    print(
        "seed\tmethod\tvalidation_ce\tutil_entropy\tdead_page_fraction\t"
        "gate_mean\troute_stability"
    )
    fixed_vals: list[float] = []
    pk_vals: list[float] = []
    pk_util_gate: dict[int, bool] = {}
    fixed_better: dict[int, bool] = {}
    pk_better: dict[int, bool] = {}

    for seed in SEEDS:
        bank = g6a.make_hidden_bank(
            model, train, g7.PAGE_STEPS, g7.PAGE_BATCH_SIZE, 50000 + seed
        )
        fixed_pages = g7.train_fixed(model, bank, projection, thresholds, seed)
        fixed_res = g7.eval_fixed(
            model, fixed_pages, eval_bank, projection, thresholds
        )
        pk_pages = train_pages_frozen_pk(model, routers[seed], bank, seed)
        pk_res = g7.eval_pk(model, pk_pages, routers[seed], eval_bank)
        fixed_stability, pk_stability = route_stability_pair(
            projection, thresholds, routers[seed], eval_bank, seed
        )
        fixed_vals.append(fixed_res.ce)
        pk_vals.append(pk_res.ce)
        pk_util_gate[seed] = (
            pk_res.entropy >= UTIL_ENTROPY_MIN and pk_res.dead <= DEAD_MAX
        )
        fixed_better[seed] = fixed_res.ce < pk_res.ce
        pk_better[seed] = pk_res.ce < fixed_res.ce
        print(
            f"{seed}\tfixed\t{fixed_res.ce:.8f}\t{fixed_res.entropy:.8f}\t"
            f"{fixed_res.dead:.8f}\t1.00000000\t{fixed_stability:.8f}"
        )
        print(
            f"{seed}\tpk_d6_balanced\t{pk_res.ce:.8f}\t{pk_res.entropy:.8f}\t"
            f"{pk_res.dead:.8f}\t{pk_res.gate_mean:.8f}\t{pk_stability:.8f}"
        )
        del bank, fixed_pages, pk_pages

    fixed_arr = np.asarray(fixed_vals, dtype=np.float64)
    pk_arr = np.asarray(pk_vals, dtype=np.float64)
    util_pass = all(pk_util_gate.values())
    fixed_mean_better = bool(fixed_arr.mean() < pk_arr.mean())
    pk_mean_better = bool(pk_arr.mean() < fixed_arr.mean())

    print()
    print("method\tvalidation_ce_mean\tvalidation_ce_sample_std\tdelta_vs_fixed_mean")
    print(
        f"fixed\t{fixed_arr.mean():.8f}\t{fixed_arr.std(ddof=1):.8f}\t+0.00000000"
    )
    print(
        f"pk_d6_balanced\t{pk_arr.mean():.8f}\t{pk_arr.std(ddof=1):.8f}\t"
        f"{pk_arr.mean() - fixed_arr.mean():+.8f}"
    )
    print()
    print(
        "pk_utilization_gate_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in pk_util_gate.items())
    )
    print(
        "fixed_beats_pk_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in fixed_better.items())
    )
    print(
        "pk_beats_fixed_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in pk_better.items())
    )
    print(f"utilization_gate_passed={str(util_pass).lower()}")
    print("resource_gate_passed=true")

    if not util_pass:
        classification = "inconclusive_anti_collapse_failed"
    elif all(pk_better.values()) and pk_mean_better:
        classification = "finite_n_uniqueness_killed"
    elif all(fixed_better.values()) and fixed_mean_better:
        classification = "fixed_factorized_supported"
    else:
        classification = "unresolved"
    print(f"g7b_classification={classification}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", required=True)
    p.add_argument("--validation-path", required=True)
    p.add_argument("--tokenizer-json", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threads", type=int, default=8)
    run(p.parse_args())
