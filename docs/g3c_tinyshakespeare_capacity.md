# G3c — Tiny Shakespeare external-capacity benchmark

## Status

**MIXED / CONSTRAINING.**

On the canonical Tiny Shakespeare corpus, the original fixed-hash control is strictly monotone for `N = 1, 4, 16, 64, 256` in all three page-training seeds. The learned prefix-balanced, reliability-ordered causal router is strictly monotone through 64 pages in all three seeds, but the `64 -> 256` step regresses slightly for seeds 8 and 9 and in the three-seed mean.

Therefore the strict learned-router 256-page extension is **not passed**. The negative edge is retained rather than tuned away.

A later paired replication, G3f, removes shape-dependent RNG coupling between page initialization and minibatch sampling. It confirms the same qualitative boundary: fixed-hash scaling remains strictly monotone through 256 pages in all three seeds, while the learned router is effectively flat from 64 to 256 and two of three seeds regress. See `docs/g3d_j_tinyshakespeare_controls.md`.

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

The capacity sweep is `N in {1, 4, 16, 64, 256}`. Maximum address width is eight bits and the full maximum-width router is allocated/executed in every capacity condition; smaller conditions change only the hard-address prefix used.

### Fixed-hash control

- one fixed `48 x 8` projection;
- eight calibrated thresholds;
- 392 resident router scalars;
- 384 projection MACs/token.

### Learned causal router

- router: `48 -> 64 -> 8` GELU MLP;
- 3,656 resident router parameters;
- 3,584 router matrix MACs/token;
- perturbation consistency from causal block-1 hidden states;
- equal composite Renyi-2 balance on raw `2, 4, 6, 8`-factor prefixes;
- confidence/discreteness pressure;
- no language-model labels, realized next-token page targets, semantic addresses, validation loss, or future-token information in router training;
- frozen factor ordering from clean-vs-perturbed hard-bit agreement on a deterministic training-only hidden-state bank.

Reliability order: `[4, 7, 6, 2, 5, 0, 3, 1]`.

## Frozen backbone

Validation CE before ParamProbe pages: `2.47749685`.

## Original fixed-hash results

| pages | seed 7 | seed 8 | seed 9 | mean ± sample std |
|---:|---:|---:|---:|---:|
| 1 | 2.46629957 | 2.46603789 | 2.46774095 | `2.46669280 ± 0.00091710` |
| 4 | 2.46591506 | 2.46554333 | 2.46549762 | `2.46565200 ± 0.00022896` |
| 16 | 2.46470086 | 2.46458505 | 2.46441248 | `2.46456613 ± 0.00014512` |
| 64 | 2.46405854 | 2.46388837 | 2.46390858 | `2.46395183 ± 0.00009296` |
| 256 | **2.46354565** | **2.46381285** | **2.46375535** | **`2.46370462 ± 0.00014064`** |

The ordering is strictly monotone in every seed.

## Original learned-router results

| pages | seed 7 | seed 8 | seed 9 | mean ± sample std |
|---:|---:|---:|---:|---:|
| 1 | 2.46629957 | 2.46603789 | 2.46774095 | `2.46669280 ± 0.00091710` |
| 4 | 2.46533078 | 2.46565932 | 2.46534231 | `2.46544414 ± 0.00018645` |
| 16 | 2.46423364 | 2.46434983 | 2.46423621 | `2.46427323 ± 0.00006635` |
| 64 | **2.46265083** | **2.46250596** | **2.46218855** | **`2.46244845 ± 0.00023645`** |
| 256 | 2.46250948 | 2.46259659 | 2.46241459 | `2.46250689 ± 0.00009103` |

Monotonicity by seed: seed 7 true, seed 8 false, seed 9 false. The mean `64 -> 256` change is `+0.00005844` CE.

At 256 pages, learned-router utilization entropy is about `0.92425`, dead-page fraction `0.00390625`, and full-address perturbation stability about `0.81735`.

## Interpretation after matched controls

G3c remains the frozen first named-corpus result. Later controls sharpen its interpretation rather than replace it:

- G3f paired training confirms strict fixed-hash monotonicity through 256 pages and an effectively flat learned 64→256 edge;
- G3g shows learned ParamProbe at 64 pages beats a resident dense adapter with nearly identical total active matrix compute;
- G3i/G3j show conventional flat MoE routing beats the present learned ParamProbe router at finite `N=64`, even with the exact same 4 KiB expert and less active matrix compute;
- G3h shows a non-monotone page-size tradeoff at fixed 1 MiB total external capacity, with 16 KiB best among 4/16/64 KiB.

The defensible claim is therefore resource-structured capacity scaling, not finite-`N` routing-quality superiority:

> On Tiny Shakespeare, inactive external page capacity can improve validation loss at fixed one-page external traffic and fixed active page compute. The fixed router is robustly monotone through 256 pages; the current learned factorized router is strong through 64 pages but plateaus at 256 and is lower-quality than a conventional flat MoE at `N=64`.

## Reproducibility

Original G3c workflow run: `33323339227`.
Artifact `g3c-tinyshakespeare-results`, id `9735553536`, SHA-256 `d1c5145932f4a824215f99c72f243c6049fa5879b9c1300b5189db3bd936167f`.

Successful environment: Python 3.12.14, NumPy 2.3.5, PyTorch 2.10.0+cpu, Ubuntu 24.04.

Reproducible workflow: `.github/workflows/g3c_tinyshakespeare.yml`.
Full later controls: `docs/g3d_j_tinyshakespeare_controls.md`.