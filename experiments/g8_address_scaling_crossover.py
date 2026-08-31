"""G8: predeclared exact address-resource scaling crossover benchmark."""
from __future__ import annotations

import math
import statistics
import time

import torch


D_MODEL = 96
D_KEY = 6
BLOCK_BYTES = 16_384
BATCH = 512
WARMUP = 5
ITERS = 20
CAPACITY_BITS = (8, 12, 14, 16, 20, 24, 30)
HIDDEN_SEED = 88_001
FACTOR_SEED = 88_002
PK_SEED = 88_003


def factor_resources(r: int) -> tuple[int, int]:
    macs = D_MODEL * r
    resident_bytes = 4 * ((D_MODEL + 1) * r)
    return macs, resident_bytes


def pk_resources(n: int) -> tuple[int, int, int]:
    root = math.isqrt(n)
    if root * root != n:
        raise ValueError("N must be a perfect square")
    macs = D_MODEL * D_KEY + D_KEY * root
    learned_bytes = 4 * ((D_MODEL * D_KEY + D_KEY) + D_KEY * root)
    bn_buffer_bytes = 56
    return macs, learned_bytes, bn_buffer_bytes


def factor_route(hidden: torch.Tensor, projection: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    bits = hidden @ projection > thresholds
    ids = torch.zeros(hidden.shape[0], dtype=torch.long)
    for i in range(bits.shape[1]):
        ids = (ids << 1) | bits[:, i].long()
    return ids


def pk_route(
    hidden: torch.Tensor,
    query_w: torch.Tensor,
    query_b: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    keys_a: torch.Tensor,
    keys_b: torch.Tensor,
) -> torch.Tensor:
    query = hidden @ query_w + query_b
    query = (query - running_mean) / torch.sqrt(running_var + 1e-5)
    qa, qb = query[:, :3], query[:, 3:]
    scores_a = qa @ keys_a.T
    scores_b = qb @ keys_b.T
    ia = scores_a.argmax(dim=-1)
    ib = scores_b.argmax(dim=-1)
    root = keys_a.shape[0]
    return ia * root + ib


def benchmark(fn) -> tuple[float, float]:
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(ITERS):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()
        samples.append((end - start) / BATCH)
    return statistics.median(samples), statistics.mean(samples)


def main() -> None:
    torch.set_num_threads(8)
    torch.manual_seed(HIDDEN_SEED)
    hidden = torch.randn(BATCH, D_MODEL, dtype=torch.float32)

    print("# ParamProbe G8: address-scaling crossover")
    print("protocol_predeclared=true")
    print(f"torch_threads={torch.get_num_threads()}")
    print(f"batch_tokens={BATCH}")
    print(f"warmup_iterations={WARMUP}")
    print(f"timed_iterations={ITERS}")
    print(f"block_bytes={BLOCK_BYTES}")
    print()
    print(
        "pages\texternal_capacity_bytes\tfactor_macs\tpk_macs\t"
        "factor_resident_bytes\tpk_resident_bytes\t"
        "factor_median_ns_per_token\tpk_median_ns_per_token\t"
        "factor_mean_ns_per_token\tpk_mean_ns_per_token"
    )

    exact_gate = True
    for r in CAPACITY_BITS:
        n = 1 << r
        root = math.isqrt(n)
        assert root * root == n

        torch.manual_seed(FACTOR_SEED + r)
        projection = torch.randn(D_MODEL, r, dtype=torch.float32)
        thresholds = torch.randn(r, dtype=torch.float32)

        torch.manual_seed(PK_SEED + r)
        query_w = torch.randn(D_MODEL, D_KEY, dtype=torch.float32)
        query_b = torch.randn(D_KEY, dtype=torch.float32)
        running_mean = torch.randn(D_KEY, dtype=torch.float32)
        running_var = torch.rand(D_KEY, dtype=torch.float32) + 0.5
        keys_a = torch.randn(root, D_KEY // 2, dtype=torch.float32)
        keys_b = torch.randn(root, D_KEY // 2, dtype=torch.float32)

        factor_ids = factor_route(hidden, projection, thresholds)
        pk_ids = pk_route(hidden, query_w, query_b, running_mean, running_var, keys_a, keys_b)
        if not bool(((factor_ids >= 0) & (factor_ids < n)).all()):
            raise RuntimeError(f"factor route out of range for N={n}")
        if not bool(((pk_ids >= 0) & (pk_ids < n)).all()):
            raise RuntimeError(f"product-key route out of range for N={n}")

        factor_macs, factor_bytes = factor_resources(r)
        pk_macs, pk_learned_bytes, pk_bn_bytes = pk_resources(n)
        pk_bytes = pk_learned_bytes + pk_bn_bytes

        expected_factor_macs = 96 * r
        expected_factor_bytes = 4 * 97 * r
        expected_pk_macs = 576 + 6 * root
        expected_pk_bytes = 4 * (582 + 6 * root) + 56
        if (factor_macs, factor_bytes, pk_macs, pk_bytes) != (
            expected_factor_macs,
            expected_factor_bytes,
            expected_pk_macs,
            expected_pk_bytes,
        ):
            raise RuntimeError("resource formula mismatch")

        if n >= 65_536 and not (factor_macs < pk_macs and factor_bytes < pk_bytes):
            exact_gate = False

        factor_med, factor_mean = benchmark(lambda: factor_route(hidden, projection, thresholds))
        pk_med, pk_mean = benchmark(
            lambda: pk_route(hidden, query_w, query_b, running_mean, running_var, keys_a, keys_b)
        )

        print(
            f"{n}\t{n * BLOCK_BYTES}\t{factor_macs}\t{pk_macs}\t"
            f"{factor_bytes}\t{pk_bytes}\t{factor_med:.2f}\t{pk_med:.2f}\t"
            f"{factor_mean:.2f}\t{pk_mean:.2f}",
            flush=True,
        )

    print()
    print(f"g8_address_scaling_resource_gate_passed={str(exact_gate).lower()}")
    if not exact_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
