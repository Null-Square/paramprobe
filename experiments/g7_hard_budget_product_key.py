"""G7: closest-prior-work hard-budget executable challenge.

Predeclared in docs/g7_hard_budget_challenge_predeclared.md and
its product-key initialization addendum before this file was created.

Compares the frozen G6 fixed factorized hash against PEER-style top-1
product-key routing using identical 256 x 16 KiB nonlinear page operators.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer

import g6a_subword_scale_lm as g6a

TOKENIZER_SHA256 = "230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb"
BACKBONE_SHA256 = "a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049"
SEEDS = (32, 33, 34)
N_PAGES = 256
PAGE_STEPS = 1920
PAGE_BATCH_SIZE = 8
CONTEXT = 128
BLOCK_BYTES = 16384
PAGE_HIDDEN = 20
RESIDUAL_SCALE = 0.15
PAGE_LR = 4e-3
WEIGHT_DECAY = 1e-4
PK_DIMS = (6, 16)
CODEBOOK_SIZE = 16


def sha256_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_assets(args: argparse.Namespace) -> tuple[Tokenizer, g6a.SubwordCausalLM]:
    tok_sha = sha256_path(args.tokenizer_json)
    model_sha = sha256_path(args.checkpoint)
    if tok_sha != TOKENIZER_SHA256:
        raise RuntimeError(f"tokenizer SHA mismatch: {tok_sha}")
    if model_sha != BACKBONE_SHA256:
        raise RuntimeError(f"backbone SHA mismatch: {model_sha}")
    tokenizer = Tokenizer.from_file(args.tokenizer_json)
    if tokenizer.get_vocab_size() != 1024:
        raise RuntimeError("frozen tokenizer vocabulary changed")
    model = g6a.SubwordCausalLM(
        vocab_size=1024,
        d_model=96,
        heads=4,
        layers=3,
        ff_dim=384,
        context=128,
        insertion_after=2,
    )
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return tokenizer, model


def init_pages(seed: int) -> g6a.PageResidualMLP:
    pages = g6a.PageResidualMLP(
        N_PAGES,
        d_model=96,
        hidden_dim=PAGE_HIDDEN,
        residual_scale=RESIDUAL_SCALE,
        block_bytes=BLOCK_BYTES,
    )
    torch.manual_seed(40000 + seed)
    torch.nn.init.normal_(pages.w1.weight, std=0.15 / math.sqrt(96))
    torch.nn.init.zeros_(pages.b1.weight)
    torch.nn.init.zeros_(pages.w2.weight)
    torch.nn.init.zeros_(pages.b2.weight)
    return pages


class ProductKeyRouter(nn.Module):
    """PEER-style top-1 Cartesian-product router with a gated page residual."""

    def __init__(self, d_key: int, seed: int) -> None:
        super().__init__()
        if d_key % 2:
            raise ValueError("d_key must be even")
        self.d_key = d_key
        self.subdim = d_key // 2
        self.query = nn.Linear(96, d_key, bias=True)
        self.query_bn = nn.BatchNorm1d(
            d_key, affine=False, eps=1e-5, momentum=0.1
        )
        self.keys1 = nn.Parameter(torch.empty(CODEBOOK_SIZE, self.subdim))
        self.keys2 = nn.Parameter(torch.empty(CODEBOOK_SIZE, self.subdim))
        torch.manual_seed(60000 + seed)
        nn.init.normal_(self.query.weight, std=1.0 / math.sqrt(96))
        nn.init.zeros_(self.query.bias)
        nn.init.normal_(self.keys1, std=1.0 / math.sqrt(self.subdim))
        nn.init.normal_(self.keys2, std=1.0 / math.sqrt(self.subdim))

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = hidden.shape[:-1]
        flat = hidden.reshape(-1, 96)
        q = self.query_bn(self.query(flat))
        q1, q2 = q[:, : self.subdim], q[:, self.subdim :]
        scores1 = q1 @ self.keys1.t()
        scores2 = q2 @ self.keys2.t()
        best1, idx1 = scores1.max(dim=-1)
        best2, idx2 = scores2.max(dim=-1)
        page_ids = idx1 * CODEBOOK_SIZE + idx2
        gate = torch.sigmoid(best1 + best2)
        return page_ids.view(shape), gate.view(shape)

    @property
    def matrix_macs_per_token(self) -> int:
        return 96 * self.d_key + 2 * CODEBOOK_SIZE * self.subdim

    @property
    def learned_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def buffer_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.buffers())


@dataclass
class EvalResult:
    ce: float
    entropy: float
    dead: float
    gate_mean: float


def utilization(ids_all: list[torch.Tensor]) -> tuple[float, float]:
    counts = torch.zeros(N_PAGES, dtype=torch.float64)
    for ids in ids_all:
        counts += torch.bincount(ids.reshape(-1).cpu(), minlength=N_PAGES).double()
    raw = counts.numpy()
    positive = raw[raw > 0]
    probs = positive / positive.sum()
    entropy = float(-(probs * np.log(probs)).sum() / np.log(N_PAGES))
    dead = float(np.mean(raw == 0))
    return entropy, dead


def train_fixed(
    model: g6a.SubwordCausalLM,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
    projection: torch.Tensor,
    thresholds: torch.Tensor,
    seed: int,
) -> g6a.PageResidualMLP:
    pages = init_pages(seed)
    opt = torch.optim.AdamW(pages.parameters(), lr=PAGE_LR, weight_decay=WEIGHT_DECAY)
    pages.train()
    for hidden, y in bank:
        with torch.no_grad():
            ids = g6a.route(hidden, projection, thresholds, 8)
        logits = model.head(model.rest(hidden + pages(ids, hidden)))
        loss = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pages


def train_pk(
    model: g6a.SubwordCausalLM,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
    seed: int,
    d_key: int,
) -> tuple[g6a.PageResidualMLP, ProductKeyRouter]:
    pages = init_pages(seed)
    router = ProductKeyRouter(d_key=d_key, seed=seed)
    params = list(pages.parameters()) + list(router.parameters())
    opt = torch.optim.AdamW(params, lr=PAGE_LR, weight_decay=WEIGHT_DECAY)
    pages.train()
    router.train()
    for hidden, y in bank:
        ids, gate = router(hidden)
        residual = pages(ids, hidden) * gate.unsqueeze(-1)
        logits = model.head(model.rest(hidden + residual))
        loss = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pages, router


def eval_fixed(
    model: g6a.SubwordCausalLM,
    pages: g6a.PageResidualMLP,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
    projection: torch.Tensor,
    thresholds: torch.Tensor,
) -> EvalResult:
    vals: list[float] = []
    ids_all: list[torch.Tensor] = []
    pages.eval()
    with torch.no_grad():
        for hidden, y in bank:
            ids = g6a.route(hidden, projection, thresholds, 8)
            logits = model.head(model.rest(hidden + pages(ids, hidden)))
            vals.append(F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1)).item())
            ids_all.append(ids)
    ent, dead = utilization(ids_all)
    return EvalResult(float(np.mean(vals)), ent, dead, 1.0)


def eval_pk(
    model: g6a.SubwordCausalLM,
    pages: g6a.PageResidualMLP,
    router: ProductKeyRouter,
    bank: list[tuple[torch.Tensor, torch.Tensor]],
) -> EvalResult:
    vals: list[float] = []
    ids_all: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    pages.eval()
    router.eval()
    with torch.no_grad():
        for hidden, y in bank:
            ids, gate = router(hidden)
            residual = pages(ids, hidden) * gate.unsqueeze(-1)
            logits = model.head(model.rest(hidden + residual))
            vals.append(F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1)).item())
            ids_all.append(ids)
            gates.append(gate.reshape(-1).cpu())
    ent, dead = utilization(ids_all)
    return EvalResult(
        float(np.mean(vals)), ent, dead, float(torch.cat(gates).mean().item())
    )


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    tokenizer, model = load_assets(args)
    train = g6a.encode_split(tokenizer, args.train_path)
    valid = g6a.encode_split(tokenizer, args.validation_path)
    if len(train) != 4_254_523 or len(valid) != 445_470:
        raise RuntimeError("tokenized split sizes changed")

    projection, thresholds = g6a.build_balanced_hash(
        model, train, max_bits=8, calibration_batches=40, batch_size=8
    )
    eval_bank = g6a.make_hidden_bank(model, valid, 40, 8, 1234)

    prototype = init_pages(SEEDS[0])
    if prototype.page_parameters != 3956 or prototype.page_payload_bytes != 15824:
        raise RuntimeError("page payload changed")
    if 2 * 96 * PAGE_HIDDEN != 3840:
        raise RuntimeError("page MAC envelope changed")
    if projection.numel() + thresholds.numel() != 776:
        raise RuntimeError("fixed router metadata changed")

    routers = {d: ProductKeyRouter(d, SEEDS[0]) for d in PK_DIMS}
    expected = {
        6: (678, 672),
        16: (1808, 1792),
    }
    for d, router in routers.items():
        if (router.learned_parameter_count, router.matrix_macs_per_token) != expected[d]:
            raise RuntimeError(f"product-key resource envelope changed for d_key={d}")
    del routers

    print("# ParamProbe G7: hard-budget product-key challenge")
    print("protocol_predeclared=true")
    print("native_resource_audit_dse_slots_per_module_token=16")
    print("native_resource_audit_scone_max_database_queries_per_token=4")
    print("native_resource_audit_peer_product_key_metadata_scaling=Theta(sqrt(N)*d_key)")
    print(f"tokenizer_sha256={sha256_path(args.tokenizer_json)}")
    print(f"backbone_sha256={sha256_path(args.checkpoint)}")
    print(f"train_tokens={len(train)}")
    print(f"validation_tokens={len(valid)}")
    print("pages=256")
    print("q_inference=1")
    print("block_bytes=16384")
    print("logical_external_bytes_per_token=16384")
    print("page_parameters=3956")
    print("page_payload_bytes=15824")
    print("active_page_macs_per_token=3840")
    print("training_steps_per_method=1920")
    print("mean_routed_assignments_per_page=7680")
    print("fixed_router_metadata_scalars=776")
    print("fixed_router_metadata_bytes=3104")
    print("fixed_router_macs_per_token=768")
    for d in PK_DIMS:
        r = ProductKeyRouter(d, SEEDS[0])
        print(f"pk_d{d}_learned_router_parameters={r.learned_parameter_count}")
        print(f"pk_d{d}_learned_router_parameter_bytes={4 * r.learned_parameter_count}")
        print(f"pk_d{d}_batchnorm_buffer_bytes={r.buffer_bytes}")
        print(f"pk_d{d}_router_macs_per_token={r.matrix_macs_per_token}")
    print()
    print("seed\tmethod\tvalidation_ce\tutil_entropy\tdead_page_fraction\tgate_mean")

    methods = ("fixed", "pk_d6", "pk_d16")
    results: dict[str, list[float]] = {m: [] for m in methods}
    by_seed: dict[int, dict[str, float]] = {}

    for seed in SEEDS:
        bank = g6a.make_hidden_bank(model, train, PAGE_STEPS, PAGE_BATCH_SIZE, 50000 + seed)
        by_seed[seed] = {}

        fixed_pages = train_fixed(model, bank, projection, thresholds, seed)
        fixed = eval_fixed(model, fixed_pages, eval_bank, projection, thresholds)
        results["fixed"].append(fixed.ce)
        by_seed[seed]["fixed"] = fixed.ce
        print(
            f"{seed}\tfixed\t{fixed.ce:.8f}\t{fixed.entropy:.8f}\t{fixed.dead:.8f}\t{fixed.gate_mean:.8f}",
            flush=True,
        )
        del fixed_pages

        for d in PK_DIMS:
            pages, router = train_pk(model, bank, seed, d)
            res = eval_pk(model, pages, router, eval_bank)
            name = f"pk_d{d}"
            results[name].append(res.ce)
            by_seed[seed][name] = res.ce
            print(
                f"{seed}\t{name}\t{res.ce:.8f}\t{res.entropy:.8f}\t{res.dead:.8f}\t{res.gate_mean:.8f}",
                flush=True,
            )
            del pages, router
        del bank

    print()
    print("method\tvalidation_ce_mean\tvalidation_ce_sample_std\tdelta_vs_fixed_mean")
    fixed_mean = float(np.mean(results["fixed"]))
    for method in methods:
        vals = np.asarray(results[method], dtype=np.float64)
        print(
            f"{method}\t{vals.mean():.8f}\t{vals.std(ddof=1):.8f}\t{vals.mean() - fixed_mean:+.8f}"
        )

    b1_better = {s: by_seed[s]["pk_d6"] < by_seed[s]["fixed"] for s in SEEDS}
    b2_better = {s: by_seed[s]["pk_d16"] < by_seed[s]["fixed"] for s in SEEDS}
    b1_mean_better = float(np.mean(results["pk_d6"])) < fixed_mean
    b2_mean_better = float(np.mean(results["pk_d16"])) < fixed_mean
    b1_kills = all(b1_better.values()) and b1_mean_better

    print()
    print("pk_d6_beats_fixed_by_seed=" + ",".join(f"{s}:{str(v).lower()}" for s, v in b1_better.items()))
    print("pk_d16_beats_fixed_by_seed=" + ",".join(f"{s}:{str(v).lower()}" for s, v in b2_better.items()))
    print(f"pk_d6_mean_better={str(b1_mean_better).lower()}")
    print(f"pk_d16_mean_better={str(b2_mean_better).lower()}")
    print("resource_gate_passed=true")
    print("g7_b1_finite_n_uniqueness_killed=" + str(b1_kills).lower())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", required=True)
    p.add_argument("--validation-path", required=True)
    p.add_argument("--tokenizer-json", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threads", type=int, default=8)
    run(p.parse_args())
