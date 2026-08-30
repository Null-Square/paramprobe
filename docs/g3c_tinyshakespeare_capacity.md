# G3c — Tiny Shakespeare external-capacity benchmark

## Status

**MIXED / CONSTRAINING.**

On the canonical Tiny Shakespeare corpus, the fixed-hash control is strictly monotone for `N = 1, 4, 16, 64, 256` in all three page-training seeds. The learned prefix-balanced, reliability-ordered causal router is strictly monotone through 64 pages in all three seeds, but the `64 -> 256` step regresses slightly for seeds 8 and 9 and in the three-seed mean.

Therefore the strict learned-router 256-page extension is **not passed**. The negative edge is retained rather than tuned away.

Successful benchmark workflow run: `33323339227`.

Result artifact: `g3c-tinyshakespeare-results`, artifact id `9735553536`, artifact SHA-256 `d1c5145932f4a824215f99c72f243c6049fa5879b9c1300b5189db3bd936167f`.

## Dataset

The run downloads the canonical Karpathy `char-rnn` Tiny Shakespeare file:

`data/tinyshakespeare/input.txt`

The workflow verifies the Git object identity before any training starts.

- bytes: `1,115,394`
- SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`
- Git blob SHA-1: `7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9`
- split: first 90% train / final 10% validation
- modeling unit: raw byte, 256-symbol vocabulary

This is a named, reproducible corpus. The model is still a small byte-level diagnostic architecture rather than the standard character-level Tiny Shakespeare architecture, so the absolute cross-entropy should not be compared directly with published character-level numbers from unrelated models.

## Frozen protocol

The successful internal insertion remains:

`embedding -> Transformer block 1 -> ParamProbe -> Transformer block 2 -> LM head`.

A fresh backbone is pretrained on Tiny Shakespeare and then frozen. Both routing experiments use exactly that same checkpoint.

The page operator is unchanged:

- `48 -> 10 -> 48` tanh residual MLP;
- 1,018 FP32 learned parameters/page;
- 4,072 learned payload bytes/page;
- one 4,096-byte physical external block/page;
- 960 active page matrix MACs/token;
- `q_inference = 1`;
- exactly 4,096 logical external parameter bytes/token.

The capacity sweep is:

`N in {1, 4, 16, 64, 256}`.

To expose 256 pages, maximum address width is eight bits. The full eight-bit router is allocated and executed in every capacity condition; smaller conditions only change how many hard address factors are used. Thus router resident size and routing compute remain fixed within each sweep.

### Fixed-hash control

- one fixed `48 x 8` projection;
- eight calibrated thresholds;
- 392 resident router scalars;
- 384 projection MACs/token;
- prefixes of the same eight-factor hash are used for all capacities.

### Learned causal router

The G3b passing protocol is extended mechanically from six to eight factors:

- router: `48 -> 64 -> 8` GELU MLP;
- 3,656 resident router parameters;
- 3,584 router matrix MACs/token;
- perturbation consistency from causal block-1 hidden states;
- equal composite Renyi-2 balance on raw `2, 4, 6, 8`-factor prefixes;
- confidence/discreteness pressure;
- no language-model labels, realized next-token page targets, semantic addresses, or validation loss in router training;
- after freezing, factors are ordered once using clean-vs-perturbed hard-bit agreement on a deterministic training-only hidden-state bank.

The resulting reliability order is:

`[4, 7, 6, 2, 5, 0, 3, 1]`.

## Environment

Successful workflow run used:

- Python `3.12.14`;
- NumPy `2.3.5`;
- PyTorch `2.10.0+cpu`;
- Ubuntu 24.04 GitHub-hosted runner.

An earlier workflow attempt failed during dependency installation because the generic PyPI PyTorch package pulled CUDA dependencies and exhausted runner disk. It failed before corpus download or model training and is not a scientific result. The successful run pins the official CPU-only PyTorch wheel.

## Frozen backbone

Validation CE before adding ParamProbe pages:

`2.47749685`.

The one-page condition improves on the frozen backbone and is identical between the fixed-hash and learned-router sweeps because no routing choice exists at `N=1`.

## Fixed-hash results

| pages | seed 7 | seed 8 | seed 9 | mean ± sample std |
|---:|---:|---:|---:|---:|
| 1 | 2.46629957 | 2.46603789 | 2.46774095 | `2.46669280 ± 0.00091710` |
| 4 | 2.46591506 | 2.46554333 | 2.46549762 | `2.46565200 ± 0.00022896` |
| 16 | 2.46470086 | 2.46458505 | 2.46441248 | `2.46456613 ± 0.00014512` |
| 64 | 2.46405854 | 2.46388837 | 2.46390858 | `2.46395183 ± 0.00009296` |
| 256 | **2.46354565** | **2.46381285** | **2.46375535** | **`2.46370462 ± 0.00014064`** |

The ordering is strictly monotone in every seed.

Hard-route utilization diagnostics are the same across page-training seeds because the router is fixed:

| pages | normalized utilization entropy | dead-page fraction |
|---:|---:|---:|
| 4 | 0.89200 | 0 |
| 16 | 0.88264 | 0 |
| 64 | 0.78244 | 0 |
| 256 | 0.72069 | 0.09375 |

The 256-page fixed hash therefore retains a monotone capacity benefit even though about 9.4% of pages receive no validation routes and the hard routing distribution is far from uniform.

## Learned-router results

| pages | seed 7 | seed 8 | seed 9 | mean ± sample std |
|---:|---:|---:|---:|---:|
| 1 | 2.46629957 | 2.46603789 | 2.46774095 | `2.46669280 ± 0.00091710` |
| 4 | 2.46533078 | 2.46565932 | 2.46534231 | `2.46544414 ± 0.00018645` |
| 16 | 2.46423364 | 2.46434983 | 2.46423621 | `2.46427323 ± 0.00006635` |
| 64 | **2.46265083** | **2.46250596** | **2.46218855** | **`2.46244845 ± 0.00023645`** |
| 256 | 2.46250948 | 2.46259659 | 2.46241459 | `2.46250689 ± 0.00009103` |

Monotonicity by page-training seed:

- seed 7: **true**;
- seed 8: **false** (`2.46250596 -> 2.46259659` at `64 -> 256`);
- seed 9: **false** (`2.46218855 -> 2.46241459` at `64 -> 256`).

The mean `64 -> 256` change is `+0.00005844` CE, so the strict learned-router benchmark criterion fails even though the regression is very small.

Routing diagnostics:

| pages | normalized utilization entropy | dead-page fraction | perturbation stability |
|---:|---:|---:|---:|
| 4 | 0.99266 | 0 | 0.98464 |
| 16 | 0.99429 | 0 | 0.95299 |
| 64 | 0.97596 | 0 | 0.90309 |
| 256 | 0.92425 | 0.00390625 | 0.81735 |

The 256-page regression coincides with a substantial drop in full-address perturbation stability, but this experiment does not establish that the stability drop is the cause.

## Comparison

Relative to the one-page condition, the learned-router mean improves by:

- 4 pages: `-0.00124867` CE;
- 16 pages: `-0.00241958` CE;
- 64 pages: `-0.00424436` CE;
- 256 pages: `-0.00418592` CE.

The 64-page learned condition corresponds to about a `0.42%` reduction in perplexity relative to its one-page control.

At equal page count, the learned-router mean is lower than the fixed-hash mean by approximately:

- 4 pages: `0.00020787` CE;
- 16 pages: `0.00029290` CE;
- 64 pages: `0.00150338` CE;
- 256 pages: `0.00119773` CE.

This is **not** an active-compute-matched router comparison: the fixed projection uses 384 routing MACs/token while the learned MLP uses 3,584. It is a routing-mechanism control, not evidence of superiority at fixed total compute across methods.

## Interpretation

G3c establishes a named-corpus version of the external-capacity effect under the strict one-page probe budget with a stable fixed router: validation loss improves monotonically through 256 external pages while the active page operator and external traffic stay fixed.

The learned causal router gives a larger benefit through 64 pages but does not preserve strict monotonicity when extended to the eight-factor / 256-page condition. This is consistent with the project's earlier address-reliability warning and should be treated as a real scaling constraint rather than hidden by averaging or post-hoc tuning.

The primary positive result is therefore narrower than a full learned-router scaling law:

> On Tiny Shakespeare, additional inactive external page capacity improves validation loss at fixed `q=1`, `B=4096`, and fixed active page compute. A fixed hash remains monotone through 256 pages; the current learned causal router remains robust through 64 pages and becomes marginally non-monotone at 256 pages.

## Next step

Do not tune the 256-page result away on this benchmark. Preserve it as the first named-corpus address-reliability boundary.

The next benchmark work should add matched controls and page-size sweeps while keeping these results frozen, including:

- dense adapter matched for active operator compute;
- a sparse/MoE-style control where feasible;
- fixed external capacity with different active bytes/token;
- fixed active bytes/token with different total external capacity;
- `B` sweeps such as 4 KiB / 16 KiB / 64 KiB;
- serialized/file-backed execution of trained language-model pages as a separate physical-storage validation.

Reproducible workflow: `.github/workflows/g3c_tinyshakespeare.yml`.
