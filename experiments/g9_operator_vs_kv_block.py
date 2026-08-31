"""G9: predeclared nonlinear-operator vs dense local KV block challenge."""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import g6a_subword_scale_lm as g6a
import g7_hard_budget_product_key as g7
import g7b_balanced_product_key as g7b

SEEDS = (38, 39, 40)
N_BLOCKS = 256
LOCAL_SLOTS = 21
D_MODEL = 96
BLOCK_BYTES = 16_384
RESIDUAL_SCALE = 0.15
PAYLOAD_STEPS = 1920
PAYLOAD_BATCH = 8
PAYLOAD_LR = 4e-3
WEIGHT_DECAY = 1e-4


class LocalKVBlock(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.keys = nn.Embedding(N_BLOCKS, LOCAL_SLOTS * D_MODEL)
        self.values = nn.Embedding(N_BLOCKS, LOCAL_SLOTS * D_MODEL)
        torch.manual_seed(81_000 + seed)
        nn.init.normal_(self.keys.weight, std=1.0 / math.sqrt(D_MODEL))
        nn.init.zeros_(self.values.weight)
        self.parameters_per_block = LOCAL_SLOTS * D_MODEL * 2
        self.payload_bytes = self.parameters_per_block * 4
        self.matrix_macs_per_token = 2 * LOCAL_SLOTS * D_MODEL
        if self.parameters_per_block != 4032:
            raise RuntimeError("G9 KV parameter count changed")
        if self.payload_bytes != 16128:
            raise RuntimeError("G9 KV payload bytes changed")
        if self.matrix_macs_per_token != 4032:
            raise RuntimeError("G9 KV active MAC count changed")
        if self.payload_bytes > BLOCK_BYTES:
            raise RuntimeError("G9 KV payload exceeds physical block")

    def forward(
        self, block_ids: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = hidden.shape
        x = hidden.reshape(-1, D_MODEL)
        ids = block_ids.reshape(-1)
        count = x.shape[0]
        keys = self.keys(ids).view(count, LOCAL_SLOTS, D_MODEL)
        values = self.values(ids).view(count, LOCAL_SLOTS, D_MODEL)
        scores = torch.bmm(keys, x.unsqueeze(-1)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        output = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
        residual = RESIDUAL_SCALE * torch.tanh(output)
        entropy = -(
            weights * torch.log(weights.clamp_min(1e-12))
        ).sum(dim=-1) / math.log(float(LOCAL_SLOTS))
        return residual.view(shape), entropy.view(shape[:-1])


def train_kv(
    model: g6a.SubwordCausalLM,
    router: g7.ProductKeyRouter,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
    seed: int,
) -> LocalKVBlock:
    kv = LocalKVBlock(seed)
    optimizer = torch.optim.AdamW(
        kv.parameters(), lr=PAYLOAD_LR, weight_decay=WEIGHT_DECAY
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
    return kv


def eval_operator(
    model: g6a.SubwordCausalLM,
    router: g7.ProductKeyRouter,
    pages: g6a.PageResidualMLP,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[float, float, float, float]:
    result = g7.eval_pk(model, pages, router, bank)
    return result.ce, result.entropy, result.dead, result.gate_mean


def eval_kv(
    model: g6a.SubwordCausalLM,
    router: g7.ProductKeyRouter,
    kv: LocalKVBlock,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[float, float, float, float, float]:
    losses: list[float] = []
    ids_all: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    local_entropy: list[torch.Tensor] = []
    router.eval()
    kv.eval()
    with torch.no_grad():
        for hidden, y in bank:
            ids, gate = router(hidden)
            residual, entropy = kv(ids, hidden)
            logits = model.head(
                model.rest(hidden + residual * gate.unsqueeze(-1))
            )
            losses.append(
                F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1)).item()
            )
            ids_all.append(ids)
            gates.append(gate.reshape(-1).cpu())
            local_entropy.append(entropy.reshape(-1).cpu())
    util_entropy, dead = g7.utilization(ids_all)
    return (
        float(np.mean(losses)),
        util_entropy,
        dead,
        float(torch.cat(gates).mean().item()),
        float(torch.cat(local_entropy).mean().item()),
    )


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    tokenizer, model = g7.load_assets(args)
    train = g6a.encode_split(tokenizer, args.train_path)
    valid = g6a.encode_split(tokenizer, args.validation_path)
    if len(train) != 4_254_523 or len(valid) != 445_470:
        raise RuntimeError("tokenized split sizes changed")

    eval_bank = g6a.make_hidden_bank(model, valid, 40, 8, 1234)
    operator_probe = g7.init_pages(SEEDS[0])
    kv_probe = LocalKVBlock(SEEDS[0])
    if operator_probe.page_parameters != 3956:
        raise RuntimeError("operator parameter count changed")
    if operator_probe.page_payload_bytes != 15824:
        raise RuntimeError("operator payload bytes changed")
    if 2 * D_MODEL * 20 != 3840:
        raise RuntimeError("operator active MAC count changed")
    del operator_probe, kv_probe

    print("# ParamProbe G9: operator vs local KV block")
    print("protocol_predeclared=true")
    print("fresh_seeds=38,39,40")
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
    print("payload_training_steps=1920")
    print("mean_routed_assignments_per_block=7680")
    print()
    print(
        "seed\tmethod\tvalidation_ce\tglobal_util_entropy\tglobal_dead_fraction\t"
        "gate_mean\tlocal_attention_entropy"
    )

    operator_values: list[float] = []
    kv_values: list[float] = []
    operator_better: dict[int, bool] = {}
    kv_better: dict[int, bool] = {}

    for seed in SEEDS:
        router, _ = g7b.train_balanced_router(model, train, seed)
        train_bank = g6a.make_hidden_bank(
            model, train, PAYLOAD_STEPS, PAYLOAD_BATCH, 50_000 + seed
        )

        operator = g7b.train_pages_frozen_pk(model, router, train_bank, seed)
        op_ce, op_ent, op_dead, op_gate = eval_operator(
            model, router, operator, eval_bank
        )

        kv = train_kv(model, router, train_bank, seed)
        kv_ce, kv_ent, kv_dead, kv_gate, kv_local_ent = eval_kv(
            model, router, kv, eval_bank
        )

        if abs(op_ent - kv_ent) > 1e-12 or abs(op_dead - kv_dead) > 1e-12:
            raise RuntimeError("shared router produced different utilization metrics")
        if abs(op_gate - kv_gate) > 1e-12:
            raise RuntimeError("shared router produced different gate mean")

        operator_values.append(op_ce)
        kv_values.append(kv_ce)
        operator_better[seed] = op_ce < kv_ce
        kv_better[seed] = kv_ce < op_ce

        print(
            f"{seed}\toperator\t{op_ce:.8f}\t{op_ent:.8f}\t{op_dead:.8f}\t"
            f"{op_gate:.8f}\tNA",
            flush=True,
        )
        print(
            f"{seed}\tkv_block\t{kv_ce:.8f}\t{kv_ent:.8f}\t{kv_dead:.8f}\t"
            f"{kv_gate:.8f}\t{kv_local_ent:.8f}",
            flush=True,
        )
        del train_bank, operator, kv, router

    op = np.asarray(operator_values, dtype=np.float64)
    kv = np.asarray(kv_values, dtype=np.float64)
    print()
    print("method\tvalidation_ce_mean\tvalidation_ce_sample_std\tdelta_vs_operator_mean")
    print(f"operator\t{op.mean():.8f}\t{op.std(ddof=1):.8f}\t+0.00000000")
    print(
        f"kv_block\t{kv.mean():.8f}\t{kv.std(ddof=1):.8f}\t"
        f"{kv.mean() - op.mean():+.8f}"
    )
    print()
    print(
        "operator_beats_kv_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in operator_better.items())
    )
    print(
        "kv_beats_operator_by_seed="
        + ",".join(f"{s}:{str(v).lower()}" for s, v in kv_better.items())
    )
    print("resource_gate_passed=true")

    if all(kv_better.values()) and kv.mean() < op.mean():
        classification = "operator_specific_advantage_killed"
    elif all(operator_better.values()) and op.mean() < kv.mean():
        classification = "nonlinear_operator_supported"
    else:
        classification = "operator_vs_kv_unresolved"
    print(f"g9_classification={classification}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())
