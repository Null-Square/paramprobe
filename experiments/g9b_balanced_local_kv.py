"""G9b: predeclared balanced local-KV challenge.

This is the single anti-collapse follow-up authorized by G9a.  The inference
mechanism and resource envelope are identical to G9; only local-KV training
changes.  Local keys receive a label-free balanced/stable partitioning phase
and are then frozen permanently before task-trained values are learned.

Protocol: docs/g9b_balanced_local_kv_predeclared.md
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import g6a_subword_scale_lm as g6a
import g7_hard_budget_product_key as g7
import g7b_balanced_product_key as g7b
import g9_operator_vs_kv_block as g9


SEEDS = (44, 45, 46)
N_BLOCKS = 256
LOCAL_SLOTS = 21
D_MODEL = 96
BLOCK_BYTES = 16_384
RESIDUAL_SCALE = 0.15

KEY_BANK_STEPS = 320
KEY_BANK_BATCH = 8
KEY_BANK_SEED = 97_531
KEY_PRETRAIN_STEPS = 600
KEY_SAMPLES_PER_BLOCK = 16
KEY_SAMPLE_SEED_BASE = 83_000
KEY_LR = 2e-3
KEY_WEIGHT_DECAY = 1e-4
KEY_NOISE_STD = 0.08
CONSISTENCY_WEIGHT = 1.0
BALANCE_WEIGHT = 0.2
CONFIDENCE_WEIGHT = 0.1

PAYLOAD_STEPS = 1_920
PAYLOAD_BATCH = 8
PAYLOAD_LR = 4e-3
PAYLOAD_WEIGHT_DECAY = 1e-4

UTIL_ENTROPY_MIN = 0.85
DEAD_MAX = 0.05
LOG_LOCAL_SLOTS = math.log(float(LOCAL_SLOTS))
TOTAL_LOCAL_SLOTS = N_BLOCKS * LOCAL_SLOTS


@dataclass(frozen=True)
class UsageStats:
    queries: int
    global_entropy: float
    dead_fraction: float
    active_slots: int
    weighted_block_entropy: float
    median_active_fraction: float
    p10_active_fraction: float
    median_max_share: float
    p90_max_share: float
    mean_softmax_entropy: float


def local_probabilities(
    kv: g9.LocalKVBlock,
    block_ids: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """Return the exact G9 21-way local softmax for selected global blocks."""
    x = hidden.reshape(-1, D_MODEL)
    ids = block_ids.reshape(-1)
    keys = kv.keys(ids).view(x.shape[0], LOCAL_SLOTS, D_MODEL)
    scores = torch.bmm(keys, x.unsqueeze(-1)).squeeze(-1)
    return torch.softmax(scores, dim=-1)


def build_key_bank(
    model: g6a.SubwordCausalLM,
    router: g7.ProductKeyRouter,
    train: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize and route the frozen 327,680-state Stage-A bank exactly once."""
    bank = g6a.make_hidden_bank(
        model, train, KEY_BANK_STEPS, KEY_BANK_BATCH, KEY_BANK_SEED
    )
    hidden = torch.cat([h.reshape(-1, D_MODEL) for h, _ in bank], dim=0)
    if hidden.shape != (327_680, D_MODEL):
        raise RuntimeError(f"G9b local-key hidden bank changed: {tuple(hidden.shape)}")
    feature_scale = hidden.std(dim=0, unbiased=False).clamp_min(1e-4)
    router.eval()
    with torch.no_grad():
        global_ids, _ = router(hidden)
    global_ids = global_ids.reshape(-1)
    if global_ids.shape != (327_680,):
        raise RuntimeError("G9b routed local-key bank shape changed")
    return hidden, global_ids, feature_scale


def pretrain_local_keys(
    kv: g9.LocalKVBlock,
    hidden: torch.Tensor,
    global_ids: torch.Tensor,
    feature_scale: torch.Tensor,
    seed: int,
) -> tuple[float, float, float, float, int]:
    """Run the predeclared 600-step label-free key-only anti-collapse phase."""
    for p in kv.values.parameters():
        p.requires_grad = False
    for p in kv.keys.parameters():
        p.requires_grad = True

    groups: list[tuple[int, torch.Tensor]] = []
    for block_id in range(N_BLOCKS):
        idx = torch.nonzero(global_ids == block_id, as_tuple=False).flatten()
        if idx.numel():
            groups.append((block_id, idx))
    if not groups:
        raise RuntimeError("G9b local-key bank has no represented global blocks")

    represented_ids = torch.tensor([block_id for block_id, _ in groups], dtype=torch.long)
    optimizer = torch.optim.AdamW(
        kv.keys.parameters(), lr=KEY_LR, weight_decay=KEY_WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(KEY_SAMPLE_SEED_BASE + seed)

    final = (float("nan"),) * 4
    kv.train()
    for _ in range(KEY_PRETRAIN_STEPS):
        rows: list[torch.Tensor] = []
        for _, idx in groups:
            local = torch.randint(
                0,
                idx.numel(),
                (KEY_SAMPLES_PER_BLOCK,),
                generator=generator,
            )
            rows.append(hidden[idx[local]])
        sampled = torch.stack(rows, dim=0)
        clean = sampled.reshape(-1, D_MODEL)
        block_ids = represented_ids[:, None].expand(
            -1, KEY_SAMPLES_PER_BLOCK
        ).reshape(-1)

        noise_a = torch.randn(
            clean.shape, generator=generator, dtype=clean.dtype
        )
        noise_b = torch.randn(
            clean.shape, generator=generator, dtype=clean.dtype
        )
        a = clean + KEY_NOISE_STD * feature_scale * noise_a
        b = clean + KEY_NOISE_STD * feature_scale * noise_b

        p_a = local_probabilities(kv, block_ids, a)
        p_b = local_probabilities(kv, block_ids, b)
        p_bar = 0.5 * (p_a + p_b)

        consistency = LOCAL_SLOTS * torch.mean((p_a - p_b) ** 2)
        marginals = p_bar.view(
            len(groups), KEY_SAMPLES_PER_BLOCK, LOCAL_SLOTS
        ).mean(dim=1)
        marginal_entropy = -(
            marginals * torch.log(marginals.clamp_min(1e-12))
        ).sum(dim=-1)
        balance = (LOG_LOCAL_SLOTS - marginal_entropy).mean()
        query_entropy = -(
            p_bar * torch.log(p_bar.clamp_min(1e-12))
        ).sum(dim=-1)
        confidence = query_entropy.mean() / LOG_LOCAL_SLOTS

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

    for p in kv.keys.parameters():
        p.requires_grad = False
    for p in kv.values.parameters():
        p.requires_grad = True
    kv.eval()
    return final + (len(groups),)


def train_values_only(
    model: g6a.SubwordCausalLM,
    router: g7.ProductKeyRouter,
    kv: g9.LocalKVBlock,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
) -> g9.LocalKVBlock:
    """Train only local values under the unchanged G9 inference formula."""
    for p in kv.keys.parameters():
        p.requires_grad = False
    for p in kv.values.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(
        kv.values.parameters(), lr=PAYLOAD_LR, weight_decay=PAYLOAD_WEIGHT_DECAY
    )
    router.eval()
    kv.train()
    for hidden, y in bank:
        with torch.no_grad():
            ids, gate = router(hidden)
        residual, _ = kv(ids, hidden)
        logits = model.head(
            model.rest(hidden + residual * gate.unsqueeze(-1))
        )
        loss = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    kv.eval()
    return kv


def local_usage_stats(
    router: g7.ProductKeyRouter,
    kv: g9.LocalKVBlock,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
) -> UsageStats:
    """Compute the exact G9a hard (global block, local slot) usage diagnostics."""
    counts = torch.zeros((N_BLOCKS, LOCAL_SLOTS), dtype=torch.float64)
    soft_entropy_sum = 0.0
    total = 0
    router.eval()
    kv.eval()

    with torch.no_grad():
        for hidden, _ in bank:
            global_ids, _ = router(hidden)
            flat_hidden = hidden.reshape(-1, D_MODEL)
            flat_ids = global_ids.reshape(-1)
            probabilities = local_probabilities(kv, flat_ids, flat_hidden)
            local_ids = probabilities.argmax(dim=-1)
            pair_ids = flat_ids.cpu() * LOCAL_SLOTS + local_ids.cpu()
            counts.view(-1).add_(
                torch.bincount(pair_ids, minlength=TOTAL_LOCAL_SLOTS).double()
            )
            entropy = -(
                probabilities
                * torch.log(probabilities.clamp_min(1e-12))
            ).sum(dim=-1) / LOG_LOCAL_SLOTS
            soft_entropy_sum += float(entropy.sum().item())
            total += int(entropy.numel())

    if total <= 0 or int(counts.sum().item()) != total:
        raise RuntimeError("G9b local-usage accounting mismatch")

    flat = counts.reshape(-1).numpy()
    positive = flat[flat > 0]
    global_probs = positive / positive.sum()
    global_entropy = float(
        -(global_probs * np.log(global_probs)).sum()
        / math.log(float(TOTAL_LOCAL_SLOTS))
    )
    dead_fraction = float(np.mean(flat == 0))
    active_slots = int(np.sum(flat > 0))

    block_totals = counts.sum(dim=1).numpy()
    observed = block_totals > 0
    observed_counts = counts.numpy()[observed]
    observed_totals = block_totals[observed]
    block_probs = observed_counts / observed_totals[:, None]
    safe_probs = np.where(block_probs > 0, block_probs, 1.0)
    block_entropy = -np.sum(
        np.where(block_probs > 0, block_probs * np.log(safe_probs), 0.0),
        axis=1,
    ) / LOG_LOCAL_SLOTS
    weighted_block_entropy = float(
        np.sum(block_entropy * observed_totals) / np.sum(observed_totals)
    )
    active_fraction = np.mean(observed_counts > 0, axis=1)
    max_share = np.max(observed_counts, axis=1) / observed_totals

    return UsageStats(
        queries=total,
        global_entropy=global_entropy,
        dead_fraction=dead_fraction,
        active_slots=active_slots,
        weighted_block_entropy=weighted_block_entropy,
        median_active_fraction=float(np.median(active_fraction)),
        p10_active_fraction=float(np.quantile(active_fraction, 0.10)),
        median_max_share=float(np.median(max_share)),
        p90_max_share=float(np.quantile(max_share, 0.90)),
        mean_softmax_entropy=float(soft_entropy_sum / total),
    )


def assert_frozen_protocol() -> None:
    """Fail loudly if imported G7/G9 constants drift away from the predeclaration."""
    if g9.N_BLOCKS != N_BLOCKS or g9.LOCAL_SLOTS != LOCAL_SLOTS:
        raise RuntimeError("G9b inherited KV shape changed")
    if g9.D_MODEL != D_MODEL or g9.BLOCK_BYTES != BLOCK_BYTES:
        raise RuntimeError("G9b inherited physical resource envelope changed")
    if not math.isclose(g9.RESIDUAL_SCALE, RESIDUAL_SCALE):
        raise RuntimeError("G9b residual scale changed")
    if g9.PAYLOAD_STEPS != PAYLOAD_STEPS or g9.PAYLOAD_BATCH != PAYLOAD_BATCH:
        raise RuntimeError("G9b inherited payload-training bank changed")
    if not math.isclose(g9.PAYLOAD_LR, PAYLOAD_LR):
        raise RuntimeError("G9b inherited payload learning rate changed")
    if not math.isclose(g9.WEIGHT_DECAY, PAYLOAD_WEIGHT_DECAY):
        raise RuntimeError("G9b inherited weight decay changed")
    inherited = (
        g7b.ROUTER_STEPS,
        g7b.ROUTER_LR,
        g7b.ROUTER_WD,
        g7b.NOISE_STD,
        g7b.CONSISTENCY_WEIGHT,
        g7b.BALANCE_WEIGHT,
        g7b.CONFIDENCE_WEIGHT,
    )
    expected = (600, 2e-3, 1e-4, 0.08, 1.0, 0.2, 0.1)
    if inherited != expected:
        raise RuntimeError("G9b inherited G7b anti-collapse hyperparameters changed")


def print_usage(seed: int, split: str, stats: UsageStats) -> None:
    print(
        f"{seed}\t{split}\t{stats.queries}\t{stats.global_entropy:.8f}\t"
        f"{stats.dead_fraction:.8f}\t{stats.active_slots}\t"
        f"{stats.weighted_block_entropy:.8f}\t"
        f"{stats.median_active_fraction:.8f}\t{stats.p10_active_fraction:.8f}\t"
        f"{stats.median_max_share:.8f}\t{stats.p90_max_share:.8f}\t"
        f"{stats.mean_softmax_entropy:.8f}",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    assert_frozen_protocol()
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    tokenizer, model = g7.load_assets(args)
    train = g6a.encode_split(tokenizer, args.train_path)
    valid = g6a.encode_split(tokenizer, args.validation_path)
    if len(train) != 4_254_523 or len(valid) != 445_470:
        raise RuntimeError("tokenized split sizes changed")

    eval_bank = g6a.make_hidden_bank(model, valid, 40, 8, 1234)
    operator_probe = g7.init_pages(SEEDS[0])
    kv_probe = g9.LocalKVBlock(SEEDS[0])
    if operator_probe.page_parameters != 3956:
        raise RuntimeError("G9b operator parameter count changed")
    if operator_probe.page_payload_bytes != 15824:
        raise RuntimeError("G9b operator payload bytes changed")
    if kv_probe.parameters_per_block != 4032 or kv_probe.payload_bytes != 16128:
        raise RuntimeError("G9b KV payload changed")
    if kv_probe.matrix_macs_per_token != 4032:
        raise RuntimeError("G9b KV MAC envelope changed")
    del operator_probe, kv_probe

    print("# ParamProbe G9b: final balanced local-KV challenge")
    print("protocol_predeclared=true")
    print("fresh_seeds=44,45,46")
    print(f"tokenizer_sha256={g7.sha256_path(args.tokenizer_json)}")
    print(f"backbone_sha256={g7.sha256_path(args.checkpoint)}")
    print("blocks=256")
    print("q_inference=1")
    print("block_bytes=16384")
    print("logical_external_bytes_per_token=16384")
    print("shared_router=balanced_product_key_d6")
    print("shared_router_parameter_bytes=2712")
    print("shared_router_batchnorm_buffer_bytes=56")
    print("shared_router_macs_per_token=672")
    print("operator_parameters_per_block=3956")
    print("operator_payload_bytes=15824")
    print("operator_padding_bytes=560")
    print("operator_active_macs_per_token=3840")
    print("kv_local_slots=21")
    print("kv_parameters_per_block=4032")
    print("kv_payload_bytes=16128")
    print("kv_padding_bytes=256")
    print("kv_active_macs_per_token=4032")
    print("key_bank_hidden_states=327680")
    print("key_bank_sampling_seed=97531")
    print("key_pretrain_steps=600")
    print("key_samples_per_represented_block=16")
    print("key_pretrain_noise_std=0.08")
    print("key_pretrain_consistency_weight=1.0")
    print("key_pretrain_balance_weight=0.2")
    print("key_pretrain_confidence_weight=0.1")
    print("key_pretrain_lr=0.002")
    print("key_pretrain_weight_decay=0.0001")
    print("payload_training_steps=1920")
    print("payload_training_batch=8")
    print("payload_training_context=128")
    print("kv_task_trainable_parameters=values_only")
    print("utilization_entropy_gate_min=0.85")
    print("dead_local_slot_fraction_gate_max=0.05")
    print()

    print(
        "record\tseed\tmetric1\tmetric2\tmetric3\tmetric4\tmetric5\tmetric6\tmetric7"
    )
    operator_values: list[float] = []
    kv_values: list[float] = []
    operator_better: dict[int, bool] = {}
    kv_better: dict[int, bool] = {}
    usage_gate: dict[int, bool] = {}
    usage_results: dict[int, tuple[UsageStats, UsageStats]] = {}

    for seed in SEEDS:
        router, _ = g7b.train_balanced_router(model, train, seed)
        if router.learned_parameter_count != 678:
            raise RuntimeError("G9b product-key router parameter count changed")
        if router.matrix_macs_per_token != 672 or router.buffer_bytes != 56:
            raise RuntimeError("G9b product-key router resource envelope changed")

        key_hidden, key_global_ids, feature_scale = build_key_bank(
            model, router, train
        )
        kv = g9.LocalKVBlock(seed)
        final = pretrain_local_keys(
            kv, key_hidden, key_global_ids, feature_scale, seed
        )
        print(
            f"key_pretrain\t{seed}\t{final[0]:.8f}\t{final[1]:.8f}\t"
            f"{final[2]:.8f}\t{final[3]:.8f}\t{final[4]}\tNA\tNA",
            flush=True,
        )
        del key_hidden, key_global_ids, feature_scale

        train_bank = g6a.make_hidden_bank(
            model, train, PAYLOAD_STEPS, PAYLOAD_BATCH, 50_000 + seed
        )
        operator = g7b.train_pages_frozen_pk(model, router, train_bank, seed)
        op_ce, op_ent, op_dead, op_gate = g9.eval_operator(
            model, router, operator, eval_bank
        )

        kv = train_values_only(model, router, kv, train_bank)
        kv_ce, kv_ent, kv_dead, kv_gate, kv_local_ent = g9.eval_kv(
            model, router, kv, eval_bank
        )

        if abs(op_ent - kv_ent) > 1e-12 or abs(op_dead - kv_dead) > 1e-12:
            raise RuntimeError("G9b shared router produced different utilization metrics")
        if abs(op_gate - kv_gate) > 1e-12:
            raise RuntimeError("G9b shared router produced different gate mean")

        train_usage = local_usage_stats(router, kv, train_bank)
        valid_usage = local_usage_stats(router, kv, eval_bank)
        usage_results[seed] = (train_usage, valid_usage)
        usage_gate[seed] = (
            train_usage.global_entropy >= UTIL_ENTROPY_MIN
            and train_usage.dead_fraction <= DEAD_MAX
        )

        operator_values.append(op_ce)
        kv_values.append(kv_ce)
        operator_better[seed] = op_ce < kv_ce
        kv_better[seed] = kv_ce < op_ce

        print(
            f"quality_operator\t{seed}\t{op_ce:.8f}\t{op_ent:.8f}\t"
            f"{op_dead:.8f}\t{op_gate:.8f}\tNA\tNA\tNA",
            flush=True,
        )
        print(
            f"quality_balanced_kv\t{seed}\t{kv_ce:.8f}\t{kv_ent:.8f}\t"
            f"{kv_dead:.8f}\t{kv_gate:.8f}\t{kv_local_ent:.8f}\tNA\tNA",
            flush=True,
        )
        del train_bank, operator, kv, router

    print()
    print(
        "usage_seed\tsplit\tqueries\tglobal_hard_entropy\tdead_local_slot_fraction\t"
        "active_local_slots\tweighted_block_entropy\tmedian_active_fraction\t"
        "p10_active_fraction\tmedian_max_slot_share\tp90_max_slot_share\t"
        "mean_softmax_entropy"
    )
    for seed in SEEDS:
        train_usage, valid_usage = usage_results[seed]
        print_usage(seed, "train", train_usage)
        print_usage(seed, "validation", valid_usage)

    op = np.asarray(operator_values, dtype=np.float64)
    kv_arr = np.asarray(kv_values, dtype=np.float64)
    print()
    print("method\tvalidation_ce_mean\tvalidation_ce_sample_std\tdelta_vs_operator_mean")
    print(f"operator\t{op.mean():.8f}\t{op.std(ddof=1):.8f}\t+0.00000000")
    print(
        f"balanced_kv\t{kv_arr.mean():.8f}\t{kv_arr.std(ddof=1):.8f}\t"
        f"{kv_arr.mean() - op.mean():+.8f}"
    )
    print()
    print(
        "local_usage_healthy_by_seed="
        + ",".join(f"{s}:{str(usage_gate[s]).lower()}" for s in SEEDS)
    )
    print(
        "operator_beats_balanced_kv_by_seed="
        + ",".join(f"{s}:{str(operator_better[s]).lower()}" for s in SEEDS)
    )
    print(
        "balanced_kv_beats_operator_by_seed="
        + ",".join(f"{s}:{str(kv_better[s]).lower()}" for s in SEEDS)
    )

    if not all(usage_gate.values()):
        classification = "anti_collapse_failed"
    elif all(kv_better.values()) and kv_arr.mean() < op.mean():
        classification = "operator_specific_advantage_killed"
    elif all(operator_better.values()) and op.mean() < kv_arr.mean():
        classification = "nonlinear_operator_supported_after_balanced_kv"
    else:
        classification = "operator_vs_balanced_kv_unresolved"
    print(f"g9b_classification={classification}")
    print("further_kv_tuning_allowed=false")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())
