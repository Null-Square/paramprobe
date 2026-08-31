# G8: predeclared address-scaling crossover challenge

Status before execution: **PREDECLARED / UNRUN**

## Purpose

G7b kills any claim that the current fixed factorized ParamProbe router is uniquely superior at finite N=256. A balanced PEER-style top-1 product-key router satisfies the same q=1 / 16 KiB page contract, uses fewer finite-N routing MACs/state, and achieves lower CE.

G8 therefore does not compare LM quality. It tests the surviving architecture-specific claim: **address-resource scaling**.

The question is whether the binary factorized router's `Theta(log N)` resident address work/state becomes materially smaller than exact two-bank product-key routing's `Theta(sqrt(N) * d_key)` work/state at larger external capacities.

No result from G8 can overturn G7b's finite-N quality result.

## Frozen routing definitions

Use `d_model=96` and the G7/G7b resource-matched product-key width `d_key=6`.

### Binary factorized router

For `N=2^r` pages:

- resident FP32 projection: `96 x r`;
- resident FP32 thresholds: `r`;
- resident routing scalars: `97r`;
- resident routing bytes: `4 * 97r`;
- matrix MACs/token: `96r`;
- exact hard address formed by r thresholded bits.

This is the current fixed balanced-hash family used in G6/G7. G8 measures the addressing kernel only; train-only threshold calibration is irrelevant to the exact resource count.

### Exact two-bank product-key router

Use the G7b `d_key=6` shape:

- query linear: `96 -> 6`;
- two subqueries of width 3;
- each codebook contains `sqrt(N)` 3D subkeys;
- exact top-1 address is the Cartesian pair of the independently best subkeys;
- query parameters: `96*6 + 6 = 582` FP32;
- codebook parameters: `6*sqrt(N)` FP32;
- learned resident routing bytes: `4 * (582 + 6*sqrt(N))`;
- non-affine BatchNorm running-state buffers: 56 bytes, reported separately and included in total resident bytes;
- query + exact subkey score MACs/token: `96*6 + 6*sqrt(N) = 576 + 6*sqrt(N)`.

No approximate nearest-neighbor index is introduced; this is the exact product-key routing mechanism used by the G7/G7b comparator.

## Capacity points

Use physical page size B=16,384 bytes only to translate N into logical external learned-store capacity. No page table is allocated.

Predeclared page counts:

- `2^8 = 256` pages -> 4 MiB external store;
- `2^12 = 4,096` pages -> 64 MiB;
- `2^14 = 16,384` pages -> 256 MiB;
- `2^16 = 65,536` pages -> 1 GiB;
- `2^20 = 1,048,576` pages -> 16 GiB;
- `2^24 = 16,777,216` pages -> 256 GiB;
- `2^30 = 1,073,741,824` pages -> 16 TiB.

All exponents are even where exact integer `sqrt(N)` is needed by the two-bank product-key construction.

## Exact expected resource crossover

The exact MAC equality solves approximately

`96 log2(N) = 576 + 6 sqrt(N)`.

For the chosen discrete points:

- N=256: factorized 768 MACs; product-key 672 MACs;
- N=4,096: factorized 1,152; product-key 960;
- N=16,384: factorized 1,344; product-key 1,344;
- N=65,536: factorized 1,536; product-key 2,112;
- N=1,048,576: factorized 1,920; product-key 6,720;
- N=16,777,216: factorized 2,304; product-key 25,152;
- N=1,073,741,824: factorized 2,880; product-key 197,184.

Resident bytes show essentially the same crossover around N=16,384 once the product-key BatchNorm buffers are included.

## Executable microbenchmark

Use pinned CPU PyTorch environment, single process.

For each N:

- deterministic synthetic hidden matrix `512 x 96`, seed 88001;
- deterministic factorized projection/threshold tensors, seed 88002;
- deterministic product-key query/codebook tensors, seed 88003;
- use FP32 throughout;
- run both routing kernels on the same hidden batch;
- warm up each kernel 5 iterations;
- benchmark 20 timed iterations with `time.perf_counter_ns()`;
- synchronize only insofar as ordinary CPU tensor operations are synchronous;
- report median and mean nanoseconds/token;
- verify every returned page id is in `[0,N)`;
- report exact analytical MACs and resident bytes alongside measured timing.

The benchmark is a CPU implementation diagnostic, not a hardware-independent performance theorem. Exact resource counts are primary; measured latency is secondary because BLAS/kernel efficiency differs by matrix shape.

## Predeclared interpretation

1. G8 **passes the address-scaling resource claim** if exact implementation resource counts match the formulas and factorized routing is strictly lower in both MACs/token and resident bytes at every predeclared point N >= 65,536.
2. Measured CPU latency does not determine pass/fail. If product-key remains faster despite more MACs/state at some points, report it without reinterpretation.
3. The N=256 G7b quality advantage remains frozen regardless of G8.
4. G8 does not establish that factorized routing has better quality at large N; such a claim would require a separate adequately trained capacity experiment.
5. No page-training or LM-quality result is inferred from the logical external capacities used in this routing-only benchmark.

No capacity point, kernel definition, timing iteration count, or pass interpretation may change after execution begins.