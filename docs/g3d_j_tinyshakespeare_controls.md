# G3d–G3j — Tiny Shakespeare controls and robustness

## Status

**FROZEN TINY SHAKESPEARE CONTROL SUITE — MIXED / CONSTRAINING, WITH A ROBUST CAPACITY EFFECT.**

The main conclusions are:

1. paired fixed-hash external capacity is strictly monotone through 256 pages in all three seeds;
2. learned causal routing improves through 64 pages and effectively plateaus at 256, with two of three paired seeds regressing;
3. a compute-matched resident dense adapter does not match learned ParamProbe at 64 pages;
4. conventional task-trained flat MoE routing **does** beat current learned ParamProbe at 64 experts, even with the exact same 4 KiB expert and less active matrix compute;
5. at fixed 1 MiB external capacity, 16 KiB pages outperform 4 KiB and 64 KiB pages in all three paired seeds;
6. pinned CPU CI passes 10/10 tests.

These results support inactive conditional capacity under a hard probe budget. They do **not** support finite-`N` routing-quality superiority over flat MoE. No further Tiny Shakespeare router tuning should be used to erase these boundaries.

## Common setup

Canonical Karpathy Tiny Shakespeare: 1,115,394 bytes, SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`, Git blob SHA-1 `7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9`, first 90% train / final 10% validation, raw-byte 256-symbol modeling. Frozen two-block Transformer backbone validation CE is `2.47749685`.

Insertion is `embedding -> block 1 -> conditional operator -> block 2 -> LM head`.

The standard ParamProbe page is `48 -> 10 -> 48`: 1,018 FP32 parameters, 4,072 learned bytes, one 4,096-byte physical page, 960 active page matrix MACs/token, `q=1`, exactly 4,096 logical external bytes/token.

## G3f — paired capacity replication

G3f explicitly pairs page-prefix initialization and training/evaluation minibatches across `N`, eliminating shape-dependent RNG coupling in the original G3c page-table construction.

| pages | fixed hash CE | learned router CE |
|---:|---:|---:|
| 1 | 2.46687950 ± 0.00048935 | 2.46687950 ± 0.00048935 |
| 4 | 2.46569404 ± 0.00032800 | 2.46534705 ± 0.00010996 |
| 16 | 2.46455893 ± 0.00030115 | 2.46422635 ± 0.00021991 |
| 64 | 2.46409338 ± 0.00034050 | **2.46244999 ± 0.00003221** |
| 256 | **2.46395130 ± 0.00037318** | 2.46245038 ± 0.00011370 |

Fixed hash is strictly monotone in every seed. Learned `64 -> 256` regresses in seeds 7 and 9, improves in seed 8, and changes by only about `+0.00000039` CE in the three-seed mean. The learned 256-page gate therefore remains a strict failure/plateau.

Run `33327039148`; artifact `9736565824`; artifact SHA-256 `77b7d9cde86c8fa0e7667a3035c397f18a9ea511dc815cc454646293f1218d41`.

## G3g — dense active-compute control

| resident adapter | matrix MACs/token | mean CE |
|---|---:|---:|
| `48 -> 10 -> 48` | 960 | 2.46710117 ± 0.00041528 |
| `48 -> 47 -> 48` | 4,512 | **2.46482733 ± 0.00007729** |

Learned ParamProbe uses 4,544 router+page matrix MACs/token and reaches `2.46244999` at 64 pages, about `0.00238` CE below the 4,512-MAC dense adapter. The gain is therefore not explained by merely adding comparable active dense compute.

Run `33327096295`; artifact `9736558447`; SHA-256 `89e2adf41d943344731c70e9785498473d08469e54f8b38a8c905a19904800f6`.

## G3d — maximum-width flat MoE quality control

A resident 256-way flat router executes 12,288 routing matrix MACs/token plus the same 960-MAC `48 -> 10 -> 48` expert. Three-seed mean CE for 1/4/16/64/256 experts is `2.46640849 / 2.46517187 / 2.46378996 / 2.46164943 / 2.46165924`. It improves strongly through 64 and has a tiny mean 64→256 regression.

This is a strong finite-`N` quality control but has `O(N)` resident router state/compute and no external-I/O claim.

Run `33326900388`; artifact `9736542318`; SHA-256 `9d2f3bbdaa4fd2624d1a5565aafa9e115914d5c5e7ecc3ca22989ccdfbb5abed`.

## G3i — total-compute-matched flat MoE

Maximum 64-way flat router: 3,072 routing MACs/token. Expert `48 -> 15 -> 48`: 1,440 MACs/token. Total = **4,512**, only 32 below learned ParamProbe's 4,544.

| experts | mean CE |
|---:|---:|
| 1 | 2.46556124 ± 0.00036420 |
| 4 | 2.46474781 ± 0.00036806 |
| 16 | 2.46347691 ± 0.00076380 |
| 64 | **2.46123072 ± 0.00015764** |

At essentially identical matrix compute, flat MoE is about `0.00122` CE lower than paired learned ParamProbe at 64 pages.

Run `33327309432`; artifact `9736630965`; SHA-256 `f3c9ea7c7c04616e6150fa4780a4c321895cef725a142bb4729dbdd170e58d13`.

## G3j — exact 4 KiB page-shape-matched flat MoE

This is the strictest finite-`N` router control:

- same exact `48 -> 10 -> 48` expert, 1,018 parameters, 4,072 bytes, 960 expert MACs/token;
- flat maximum 64-way router, 3,072 router MACs/token;
- total **4,032 MACs/token**, 512 fewer than learned ParamProbe.

| experts | mean CE |
|---:|---:|
| 1 | 2.46640849 ± 0.00082645 |
| 4 | 2.46516396 ± 0.00063630 |
| 16 | 2.46373103 ± 0.00052195 |
| 64 | **2.46182543 ± 0.00038872** |

The page-shape-matched flat MoE beats paired learned ParamProbe at 64 by about `0.00062456` CE while using less active matrix compute. Thus the current factorized learned router is not quality-superior to conventional flat routing at `N=64`.

The resource distinction remains: flat routing has resident metadata linear in maximum `N`, and these controls keep experts resident. They do not satisfy the intended bounded/scalable routing plus hard external-parameter-probe contract.

Run `33327465444`; artifact `9736674274`; SHA-256 `7e219908484f990c0b06bff290cb01f87b242a51770959bac25758973ef28e1f`.

## G3h — paired page granularity

At fixed exactly 1 MiB physical external capacity:

| B | pages | hidden width | active page MACs/token | mean CE |
|---:|---:|---:|---:|---:|
| 4 KiB | 256 | 10 | 960 | 2.46392645 ± 0.00038834 |
| 16 KiB | 64 | 41 | 3,936 | **2.46253466 ± 0.00023699** |
| 64 KiB | 16 | 168 | 16,128 | 2.46337525 ± 0.00011163 |

16 KiB is best in all three seeds. More bytes/probe are not automatically better; page/operator granularity is a genuine co-design variable.

Run `33327170286`; artifact `9736590384`; SHA-256 `375c62eba62a3f42cd379fddffba429a29162ff0d624fd937db162f242b6b3e1`.

## CI

Pinned CPU CI under Python 3.12 / PyTorch 2.10 CPU passes **10/10 tests**. Successful run: `33327415104`.

## Frozen interpretation

**Supported:** inactive external page capacity can improve Tiny Shakespeare validation loss under `q=1`, fixed 4 KiB traffic, and fixed active page compute. The paired fixed-router curve is monotone through 256 pages.

**Not supported:** clean learned-router scaling through 256 pages; the current learned factorized router plateaus at the eight-factor edge.

**Not supported:** finite-`N` routing-quality superiority over conventional flat MoE. The exact page-shape-matched flat 64-way MoE is lower-loss with fewer active matrix MACs.

Accordingly, ParamProbe's strongest empirical contribution at this stage is **capacity scaling under a hard external-parameter probe/resource structure**, not beating MoE on quality. Tiny Shakespeare is now frozen. The next work should move to a second named language corpus and then serialized/file-backed execution of trained LM pages.