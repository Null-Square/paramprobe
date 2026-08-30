# G3c — Tiny Shakespeare external-capacity benchmark

## Status

**FROZEN / MIXED / CONSTRAINING.**

On canonical Tiny Shakespeare, the original fixed-hash control is strictly monotone for `N = 1, 4, 16, 64, 256` in all three page-training seeds. The learned prefix-balanced, reliability-ordered causal router is strictly monotone through 64 pages in all three seeds but is slightly non-monotone at `64 -> 256`.

The later paired G3f replication removes shape-dependent RNG coupling between page initialization and minibatch sampling and confirms the same qualitative boundary: fixed-hash scaling remains strictly monotone through 256 pages in all three seeds, while the learned router is effectively flat from 64 to 256 and two of three seeds regress. The Tiny Shakespeare stage is therefore frozen rather than tuned further. See `docs/g3d_j_tinyshakespeare_controls.md`.

## Dataset

Canonical Karpathy `char-rnn` Tiny Shakespeare:

- bytes: `1,115,394`
- SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`
- Git blob SHA-1: `7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9`
- first 90% train / final 10% validation
- raw-byte 256-symbol modeling

The model is a small byte-level diagnostic architecture; absolute CE should not be compared directly with unrelated character-level Tiny Shakespeare models.

## Frozen protocol

Insertion:

`embedding -> Transformer block 1 -> ParamProbe -> Transformer block 2 -> LM head`.

Page operator:

- `48 -> 10 -> 48` residual MLP;
- 1,018 FP32 learned parameters/page;
- 4,072 learned payload bytes/page;
- 4,096-byte physical block;
- 960 active page matrix MACs/token;
- `q=1`;
- exactly 4,096 logical external parameter bytes/token.

Maximum address width is eight bits and the maximum-width router is allocated/executed across the entire capacity sweep.

Fixed hash: 392 resident router scalars and 384 projection MACs/token.

Learned router: `48 -> 64 -> 8`, 3,656 resident parameters, 3,584 router matrix MACs/token, causal hidden-state perturbation consistency, composite Renyi-2 prefix balancing on 2/4/6/8 factors, confidence pressure, and training-only perturbation-stability factor ordering. No semantic address labels, realized next-token page targets, validation loss, or future-token information are used for routing supervision.

Reliability order: `[4, 7, 6, 2, 5, 0, 3, 1]`.

Frozen backbone validation CE: `2.47749685`.

## Original G3c results

| pages | fixed hash mean ± std | learned router mean ± std |
|---:|---:|---:|
| 1 | `2.46669280 ± 0.00091710` | `2.46669280 ± 0.00091710` |
| 4 | `2.46565200 ± 0.00022896` | `2.46544414 ± 0.00018645` |
| 16 | `2.46456613 ± 0.00014512` | `2.46427323 ± 0.00006635` |
| 64 | `2.46395183 ± 0.00009296` | **`2.46244845 ± 0.00023645`** |
| 256 | **`2.46370462 ± 0.00014064`** | `2.46250689 ± 0.00009103` |

Fixed hash is strictly monotone in every seed. Learned routing regresses at `64 -> 256` in seeds 8 and 9; mean change is `+0.00005844` CE.

At 256 pages, learned-router utilization entropy is about `0.92425`, dead-page fraction `0.00390625`, and full-address perturbation stability about `0.81735`.

## Interpretation after paired/matched controls

Later controls sharpen rather than erase G3c:

- **G3f:** strict pairing confirms fixed-hash monotonicity through 256 and an effectively flat learned 64→256 edge;
- **G3g:** learned ParamProbe at 64 beats a resident dense adapter with nearly identical total active matrix compute;
- **G3i/G3j:** conventional flat MoE routing beats the present learned ParamProbe router at finite `N=64`, including with the exact same 4 KiB expert and less active matrix compute;
- **G3h:** page size has a non-monotone tradeoff at fixed 1 MiB total external capacity, with 16 KiB best among 4/16/64 KiB.

The defensible claim is therefore resource-structured capacity scaling, not finite-`N` routing-quality superiority:

> On Tiny Shakespeare, inactive external page capacity can improve validation loss at fixed one-page external traffic and fixed active page compute. The fixed router is robustly monotone through 256 pages; the current learned factorized router is strong through 64 pages but plateaus at 256 and is lower-quality than a conventional flat MoE at `N=64`.

## Reproducibility

Original G3c workflow run: `33323339227`.
Artifact `g3c-tinyshakespeare-results`, id `9735553536`, SHA-256 `d1c5145932f4a824215f99c72f243c6049fa5879b9c1300b5189db3bd936167f`.

Successful environment: Python 3.12.14, NumPy 2.3.5, PyTorch 2.10.0+cpu, Ubuntu 24.04.

Workflow: `.github/workflows/g3c_tinyshakespeare.yml`.
Full frozen control suite: `docs/g3d_j_tinyshakespeare_controls.md`.