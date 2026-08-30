"""G4b: trained Tiny Shakespeare pages through file-backed parameter probes.

This experiment connects the trained G3 LM operator to the actual fixed-block
storage backends. It deliberately separates two questions:

1. Functional equivalence: do resident, pread-backed, and Linux O_DIRECT-backed
   executions of the same trained page produce the same residual/LM loss while
   accounting exactly one 4 KiB probe per token?
2. Storage scaling: if the same trained 64-page table is mirrored across larger
   backing files and physical reads are spread across the whole file, do process
   RSS and per-probe traffic remain bounded as file size grows?

The mirrored-file sweep is a *systems* experiment only. Mirrored copies are not
additional learned capacity and no quality-scaling claim is made from them.
GitHub-hosted runner storage is also not publication hardware; device/controller
caches and virtualization remain uncontrolled even with O_DIRECT.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

import g3a_internal_fixed_hash_lm as base
import g3b_prefix_reliable_lm as learned
import g3f_paired_capacity_lm as paired
from paramprobe.page_mlp import PageMLPLayout, apply_page_mlp_block, encode_page_mlp
from paramprobe.store import DirectIOParameterStore, FileParameterStore

MIB = 1024 * 1024


def current_rss_bytes() -> int:
    """Best-effort current process RSS from Linux procfs."""
    try:
        fields = Path("/proc/self/statm").read_text().split()
        return int(fields[1]) * os.sysconf("SC_PAGESIZE")
    except Exception:
        # ru_maxrss is peak rather than current RSS, but is a useful fallback.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def encode_trained_pages(pages: base.PageResidualMLP, block_bytes: int) -> list[bytes]:
    blocks: list[bytes] = []
    with torch.no_grad():
        for page_id in range(pages.num_pages):
            w1 = pages.w1.weight[page_id].reshape(pages.hidden_dim, pages.d_model)
            b1 = pages.b1.weight[page_id]
            w2 = pages.w2.weight[page_id].reshape(pages.d_model, pages.hidden_dim)
            b2 = pages.b2.weight[page_id]
            blocks.append(
                encode_page_mlp(
                    w1.detach().cpu().numpy(),
                    b1.detach().cpu().numpy(),
                    w2.detach().cpu().numpy(),
                    b2.detach().cpu().numpy(),
                    block_bytes,
                )
            )
    return blocks


def write_blocks(path: Path, blocks: list[bytes], copies: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb", buffering=MIB) as handle:
        for _ in range(copies):
            for block in blocks:
                handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())


def external_residual_from_blocks(
    blocks: list[bytes],
    ids: torch.Tensor,
    hidden: torch.Tensor,
    layout: PageMLPLayout,
    residual_scale: float,
) -> torch.Tensor:
    flat_ids = ids.reshape(-1).detach().cpu().numpy()
    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).detach().cpu().numpy()
    output = np.empty_like(flat_hidden, dtype=np.float32)
    for row, (page_id, vector) in enumerate(zip(flat_ids, flat_hidden)):
        raw = apply_page_mlp_block(blocks[int(page_id)], vector, layout)
        output[row] = residual_scale * np.tanh(raw)
    return torch.from_numpy(output).reshape_as(hidden)


def external_residual_from_store(
    store,
    physical_ids: np.ndarray,
    hidden: torch.Tensor,
    layout: PageMLPLayout,
    residual_scale: float,
) -> torch.Tensor:
    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).detach().cpu().numpy()
    if physical_ids.shape != (flat_hidden.shape[0],):
        raise ValueError("physical id count does not match token count")
    output = np.empty_like(flat_hidden, dtype=np.float32)
    for row, (page_id, vector) in enumerate(zip(physical_ids, flat_hidden)):
        block = store.read_block(int(page_id))
        raw = apply_page_mlp_block(block, vector, layout)
        output[row] = residual_scale * np.tanh(raw)
    return torch.from_numpy(output).reshape_as(hidden)


def deterministic_eval_batch(
    validation: torch.Tensor,
    model: base.TwoBlockByteLM,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    return base.get_batch(validation, model.context, batch_size)


def physical_ids_for_mirrors(
    base_ids: np.ndarray,
    copies: int,
    seed: int,
) -> np.ndarray:
    if copies <= 0:
        raise ValueError("copies must be positive")
    if copies == 1:
        return base_ids.astype(np.int64, copy=True)
    rng = np.random.default_rng(seed)
    banks = rng.integers(0, copies, size=base_ids.shape[0], dtype=np.int64)
    num_base_pages = int(base_ids.max()) + 1
    # The caller always uses a dense 0..N-1 base page table. Use the explicit
    # table width supplied through max(base_id)+1 only if every page is observed
    # would be unsafe, so this helper is replaced by the caller's known width.
    return banks * num_base_pages + base_ids


def spread_physical_ids(
    base_ids: np.ndarray,
    base_pages: int,
    copies: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if copies == 1:
        banks = np.zeros(base_ids.shape[0], dtype=np.int64)
    else:
        banks = rng.integers(0, copies, size=base_ids.shape[0], dtype=np.int64)
    return banks * base_pages + base_ids.astype(np.int64, copy=False)


def advise_drop_cache(store: FileParameterStore) -> bool:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    try:
        os.posix_fadvise(store._fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False


def benchmark_store(
    store,
    physical_ids: np.ndarray,
    vectors: np.ndarray,
    layout: PageMLPLayout,
    residual_scale: float,
) -> dict[str, float]:
    store.stats.reset()
    rss_before = current_rss_bytes()
    checksum = 0.0
    start = time.perf_counter()
    for block_id, vector in zip(physical_ids, vectors):
        block = store.read_block(int(block_id))
        raw = apply_page_mlp_block(block, vector, layout)
        checksum += float(np.sum(residual_scale * np.tanh(raw), dtype=np.float64))
    elapsed = time.perf_counter() - start
    rss_after = current_rss_bytes()
    probes = len(physical_ids)
    expected_bytes = probes * store.block_bytes
    if store.stats.probes != probes:
        raise AssertionError(f"probe mismatch {store.stats.probes} != {probes}")
    if store.stats.bytes_read != expected_bytes:
        raise AssertionError(f"byte mismatch {store.stats.bytes_read} != {expected_bytes}")
    return {
        "elapsed_s": elapsed,
        "us_probe_e2e": elapsed * 1e6 / probes,
        "mib_s_e2e": (expected_bytes / MIB) / elapsed,
        "rss_before_mib": rss_before / MIB,
        "rss_after_mib": rss_after / MIB,
        "rss_delta_mib": (rss_after - rss_before) / MIB,
        "checksum": checksum,
        "probes": float(store.stats.probes),
        "bytes_read": float(store.stats.bytes_read),
    }


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

    router = learned.train_router(model, train, args)
    order, factor_stability = learned.reliability_order(model, router, train, args)
    pages = paired.train_learned_pages(
        model,
        router,
        train,
        args.used_bits,
        order,
        args,
        args.page_seed,
    )
    pages.eval()
    model.eval()
    router.eval()

    blocks = encode_trained_pages(pages, args.block_bytes)
    page_store = Path(args.store_path)
    write_blocks(page_store, blocks)
    layout = PageMLPLayout(args.d_model, pages.hidden_dim, args.d_model)

    x, y = deterministic_eval_batch(
        validation, model, args.equiv_batch_size, args.eval_seed
    )
    with torch.no_grad():
        hidden = model.first(x)
        ids = learned.route(router, hidden, args.used_bits, order)
        resident_residual = pages(ids, hidden)
        resident_logits = model.head(model.rest(hidden + resident_residual))
        resident_ce = F.cross_entropy(
            resident_logits.reshape(-1, 256), y.reshape(-1)
        ).item()

    encoded_residual = external_residual_from_blocks(
        blocks, ids, hidden, layout, pages.residual_scale
    )
    encoded_logits = model.head(model.rest(hidden + encoded_residual))
    encoded_ce = F.cross_entropy(
        encoded_logits.reshape(-1, 256), y.reshape(-1)
    ).item()
    encoded_max_abs = float((resident_residual - encoded_residual).abs().max().item())

    base_ids = ids.reshape(-1).detach().cpu().numpy().astype(np.int64)
    expected_probes = int(base_ids.shape[0])
    expected_bytes = expected_probes * args.block_bytes

    with FileParameterStore(page_store, args.block_bytes) as pread_store:
        pread_residual = external_residual_from_store(
            pread_store,
            base_ids,
            hidden,
            layout,
            pages.residual_scale,
        )
        pread_stats = (pread_store.stats.probes, pread_store.stats.bytes_read)
    pread_logits = model.head(model.rest(hidden + pread_residual))
    pread_ce = F.cross_entropy(
        pread_logits.reshape(-1, 256), y.reshape(-1)
    ).item()
    pread_max_abs = float((resident_residual - pread_residual).abs().max().item())

    direct_supported = True
    direct_error = ""
    direct_ce = float("nan")
    direct_max_abs = float("nan")
    direct_stats = (0, 0)
    try:
        with DirectIOParameterStore(page_store, args.block_bytes) as direct_store:
            direct_residual = external_residual_from_store(
                direct_store,
                base_ids,
                hidden,
                layout,
                pages.residual_scale,
            )
            direct_stats = (direct_store.stats.probes, direct_store.stats.bytes_read)
        direct_logits = model.head(model.rest(hidden + direct_residual))
        direct_ce = F.cross_entropy(
            direct_logits.reshape(-1, 256), y.reshape(-1)
        ).item()
        direct_max_abs = float((resident_residual - direct_residual).abs().max().item())
    except (OSError, ValueError, NotImplementedError) as exc:
        direct_supported = False
        direct_error = f"{type(exc).__name__}:{exc}".replace("\n", " ")

    if pread_stats != (expected_probes, expected_bytes):
        raise AssertionError("pread equivalence probe accounting mismatch")
    if direct_supported and direct_stats != (expected_probes, expected_bytes):
        raise AssertionError("direct equivalence probe accounting mismatch")
    if encoded_max_abs > args.max_abs_tolerance or pread_max_abs > args.max_abs_tolerance:
        raise AssertionError("serialized page output exceeded equivalence tolerance")
    if direct_supported and direct_max_abs > args.max_abs_tolerance:
        raise AssertionError("direct page output exceeded equivalence tolerance")

    print("# ParamProbe G4b: trained LM file-backed probe validation")
    print(f"corpus_bytes={len(raw)}")
    print(f"trained_pages={pages.num_pages}")
    print(f"used_bits={args.used_bits}")
    print(f"block_bytes={args.block_bytes}")
    print(f"page_payload_bytes={pages.page_payload_bytes}")
    print(f"q_inference=1")
    print(f"equivalence_tokens={expected_probes}")
    print(f"equivalence_expected_external_bytes={expected_bytes}")
    print("factor_stability=" + ",".join(f"{v:.8f}" for v in factor_stability.tolist()))
    print("reliability_order=" + ",".join(map(str, order.tolist())))
    print(f"resident_ce={resident_ce:.8f}")
    print(f"encoded_ce={encoded_ce:.8f}")
    print(f"encoded_residual_max_abs_error={encoded_max_abs:.10g}")
    print(f"pread_ce={pread_ce:.8f}")
    print(f"pread_residual_max_abs_error={pread_max_abs:.10g}")
    print(f"pread_probes={pread_stats[0]}")
    print(f"pread_bytes_read={pread_stats[1]}")
    print(f"direct_supported={str(direct_supported).lower()}")
    print(f"direct_error={direct_error}")
    if direct_supported:
        print(f"direct_ce={direct_ce:.8f}")
        print(f"direct_residual_max_abs_error={direct_max_abs:.10g}")
        print(f"direct_probes={direct_stats[0]}")
        print(f"direct_bytes_read={direct_stats[1]}")

    # Use one deterministic hidden-state/vector bank for every store size. Reads
    # are remapped to mirrored copies of the same trained pages, so the operator
    # checksum should be invariant up to accumulation order/roundoff.
    torch.manual_seed(args.sweep_hidden_seed)
    sweep_x, _ = base.get_batch(validation, model.context, args.sweep_batch_size)
    with torch.no_grad():
        sweep_hidden = model.first(sweep_x)
        sweep_ids = learned.route(router, sweep_hidden, args.used_bits, order)
    sweep_base_ids = sweep_ids.reshape(-1).cpu().numpy().astype(np.int64)
    sweep_vectors = sweep_hidden.reshape(-1, args.d_model).cpu().numpy().astype(np.float32)
    if args.sweep_probes < len(sweep_base_ids):
        sweep_base_ids = sweep_base_ids[: args.sweep_probes]
        sweep_vectors = sweep_vectors[: args.sweep_probes]
    elif args.sweep_probes > len(sweep_base_ids):
        reps = math.ceil(args.sweep_probes / len(sweep_base_ids))
        sweep_base_ids = np.tile(sweep_base_ids, reps)[: args.sweep_probes]
        sweep_vectors = np.tile(sweep_vectors, (reps, 1))[: args.sweep_probes]

    print()
    print("# Mirrored trained-page systems sweep (not a learned-capacity experiment)")
    print("backend\tstore_pages\tstore_MiB\tpass\tcache_advice\tprobes\tbytes_per_probe\tus_probe_e2e\tMiB_s_e2e\trss_before_MiB\trss_after_MiB\trss_delta_MiB\tchecksum")

    for store_pages in args.store_pages:
        if store_pages < pages.num_pages or store_pages % pages.num_pages:
            raise ValueError("every store_pages value must be a multiple of trained_pages")
        copies = store_pages // pages.num_pages
        path = page_store.with_name(f"{page_store.stem}_{store_pages}{page_store.suffix}")
        write_blocks(path, blocks, copies=copies)
        physical_ids = spread_physical_ids(
            sweep_base_ids, pages.num_pages, copies, args.sweep_address_seed + store_pages
        )

        with FileParameterStore(path, args.block_bytes) as store:
            advised = advise_drop_cache(store)
            for pass_id in (1, 2):
                result = benchmark_store(
                    store, physical_ids, sweep_vectors, layout, pages.residual_scale
                )
                print(
                    f"pread\t{store_pages}\t{path.stat().st_size / MIB:.6f}\t{pass_id}\t"
                    f"{str(advised if pass_id == 1 else False).lower()}\t{int(result['probes'])}\t"
                    f"{args.block_bytes}\t{result['us_probe_e2e']:.3f}\t{result['mib_s_e2e']:.3f}\t"
                    f"{result['rss_before_mib']:.3f}\t{result['rss_after_mib']:.3f}\t"
                    f"{result['rss_delta_mib']:.3f}\t{result['checksum']:.9f}"
                )

        if direct_supported:
            with DirectIOParameterStore(path, args.block_bytes) as store:
                for pass_id in (1, 2):
                    result = benchmark_store(
                        store, physical_ids, sweep_vectors, layout, pages.residual_scale
                    )
                    print(
                        f"odirect\t{store_pages}\t{path.stat().st_size / MIB:.6f}\t{pass_id}\tfalse\t"
                        f"{int(result['probes'])}\t{args.block_bytes}\t{result['us_probe_e2e']:.3f}\t"
                        f"{result['mib_s_e2e']:.3f}\t{result['rss_before_mib']:.3f}\t"
                        f"{result['rss_after_mib']:.3f}\t{result['rss_delta_mib']:.3f}\t"
                        f"{result['checksum']:.9f}"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-path", type=str, default=None)
    parser.add_argument("--corpus-bytes", type=int, default=1_200_000)
    parser.add_argument("--checkpoint", type=str, default=".cache/paramprobe_g4b_base.pt")
    parser.add_argument("--retrain-backbone", action="store_true")
    parser.add_argument("--backbone-seed", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--pretrain-batch-size", type=int, default=32)
    parser.add_argument("--pretrain-lr", type=float, default=3e-3)
    parser.add_argument("--max-bits", type=int, default=8)
    parser.add_argument("--used-bits", type=int, default=6)
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
    parser.add_argument("--balance-prefixes", type=str, default="2,4,6,8")
    parser.add_argument("--order-hidden-batches", type=int, default=40)
    parser.add_argument("--order-hidden-batch-size", type=int, default=24)
    parser.add_argument("--order-hidden-seed", type=int, default=97531)
    parser.add_argument("--order-noise-seed", type=int, default=86420)
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--page-init-seed", type=int, default=40000)
    parser.add_argument("--page-batch-seed", type=int, default=50000)
    parser.add_argument("--page-seed", type=int, default=7)
    parser.add_argument("--page-steps", type=int, default=180)
    parser.add_argument("--page-batch-size", type=int, default=24)
    parser.add_argument("--page-lr", type=float, default=4e-3)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--equiv-batch-size", type=int, default=4)
    parser.add_argument("--max-abs-tolerance", type=float, default=2e-6)
    parser.add_argument("--store-path", type=str, default=".cache/g4b_trained_pages.bin")
    parser.add_argument("--store-pages", type=int, nargs="+", default=[64, 1024, 16384, 65536])
    parser.add_argument("--sweep-probes", type=int, default=4096)
    parser.add_argument("--sweep-batch-size", type=int, default=16)
    parser.add_argument("--sweep-hidden-seed", type=int, default=112233)
    parser.add_argument("--sweep-address-seed", type=int, default=445566)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())