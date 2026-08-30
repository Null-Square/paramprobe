# G1a result: collision-limited capacity at fixed probe cost

Date: 2026-08-30

## Purpose

Before introducing a learned router, isolate the most basic ParamProbe claim: useful learned state can grow on external storage while per-query parameter I/O and resident routing state remain fixed.

This experiment intentionally uses a stateless deterministic router so routing optimization cannot explain the result.

## Construction

- Associations: `A = 16,384`
- Target dimension: `32`
- Targets: independent Gaussian vectors, normalized per coordinate
- Router: stateless SplitMix64 of integer key
- Active probes: `q = 1`
- Block size: `B = 4,096` bytes
- Logical external traffic/query: exactly `4,096` bytes
- Resident learned router metadata: `0` bytes
- Page value: least-squares-optimal mean target of all keys hashed to that page

For `N` external pages, page id is the low `log2(N)` bits of the same fixed 64-bit hash. Thus the routing algorithm and its state do not grow during the sweep.

## Analytic prediction

If a page contains `k` independent zero-mean unit-variance targets, its optimal stored value is their sample mean and the expected total squared residual in that page is `k - 1` per coordinate.

Let `O` be the number of occupied pages. Conditioned on occupancy,

`E[MSE | O] = 1 - O/A`.

For uniform hashing,

`E[O] = N * (1 - (1 - 1/N)^A)`,

so

`E[MSE] = 1 - (N/A) * (1 - (1 - 1/N)^A)`.

This gives a parameter-free prediction for the scaling curve.

## Result

| Pages | Logical external capacity | Empirical MSE | Occupancy floor | Uniform-hash expectation |
|---:|---:|---:|---:|---:|
| 64 | 0.25 MiB | 0.996204 | 0.996094 | 0.996094 |
| 256 | 1 MiB | 0.984556 | 0.984375 | 0.984375 |
| 1,024 | 4 MiB | 0.937351 | 0.937500 | 0.937500 |
| 4,096 | 16 MiB | 0.754802 | 0.754272 | 0.754577 |
| 16,384 | 64 MiB | 0.370877 | 0.369568 | 0.367868 |
| 65,536 | 256 MiB | 0.119304 | 0.118713 | 0.115197 |
| 262,144 | 1 GiB | 0.030685 | 0.030823 | 0.030607 |

The empirical curve closely follows the analytic collision prediction across a 4096x external-capacity sweep, while `qB` remains exactly 4 KiB/query.

## Interpretation

This is **not evidence that language-model loss will improve with external capacity**. The construction is an associative memory with a stateless router and closed-form page values. It establishes a narrower point:

> Fixed external parameter traffic does not itself impose a fixed learned-capacity ceiling. A function family can gain useful independent learned state through additional inactive pages while its per-query parameter traffic remains constant.

That removes one possible failure mode of the research premise.

## What this falsifies / does not falsify

G1a would have failed if collision-limited error did not track external page count under fixed probe cost, indicating a mistake in the resource abstraction or implementation. It passed.

It does not test:

- learned routing;
- gradient starvation / page utilization;
- compositional generalization;
- language modeling;
- physical cold-storage latency.

Those remain the substantive risks.

## Next gate: G1b

Replace the stateless known-key router with a fixed-size learned controller and a **fixed maximum address width allocated once for the whole sweep**. Vary usable page capacity by masking/merging address bits, so controller parameters and active routing compute remain literally constant across `N`.

This avoids a subtle confound: a naively resized factorized router has `O(log N)` metadata, which is asymptotically small but is not strictly constant RAM in a controlled scaling experiment.
