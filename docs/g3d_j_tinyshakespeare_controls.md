# G3d–G3j — Tiny Shakespeare controls and robustness

## Status

**MIXED, WITH A STRONGER CAPACITY RESULT AND A STRONGER BASELINE CHALLENGE.**

This sequence freezes the original G3c Tiny Shakespeare result and tests the main reviewer-facing alternatives and robustness concerns without tuning the G3c learned-router boundary.

The main conclusions so far are:

1. the fixed-hash external-capacity curve remains strictly monotone through 256 pages under a stricter paired training protocol;
2. the learned causal router again improves through 64 pages and is effectively flat / slightly non-monotone at 256 pages, so the address-reliability boundary survives pairing;
3. a resident dense adapter matched to learned ParamProbe's total active matrix compute does not match its 64-page validation loss;
4. a conventional task-trained flat MoE can beat learned ParamProbe at 64 experts, including at essentially matched total active matrix compute, but it uses flat `O(N)` resident routing state and is not a scalable hard-resident-memory baseline;
5. at fixed 1 MiB physical external capacity, a paired page-granularity sweep favors 16 KiB pages over both 4 KiB and 64 KiB pages in all three seeds.

These results strengthen the claim that inactive conditional capacity is useful, but they do **not** support a claim that the current ParamProbe router is better than conventional MoE routing at finite `N`.

## Common dataset and backbone

All experiments use the canonical Karpathy `char-rnn` Tiny Shakespeare file:

- bytes: `1,115,394`;
- SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`;
- Git blob SHA-1: `7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9`;
- first 90% train / final 10% validation;
- byte-level 256-symbol modeling;
- fresh deterministic two-block Transformer backbone for each workflow;
- frozen-backbone validation CE: `2.47749685`.

The successful ParamProbe insertion remains

`embedding -> Transformer block 1 -> conditional operator -> Transformer block 2 -> LM head`.

Unless explicitly stated otherwise, the ParamProbe page operator is `48 -> 10 -> 48`, with 1,018 FP32 parameters, 4,072 learned payload bytes, 960 active page matrix MACs/token, `q=1`, and a 4,096-byte physical page.

## G3f — paired capacity replication

### Why this replication exists

G3c reset the page-training seed before constructing each differently sized page table. Because module initialization consumes a shape-dependent amount of RNG state, the subsequent minibatch stream was not exactly paired across `N`.

G3f removes that nuisance variable without changing the architecture or training schedule:

- each page table is explicitly reinitialized after construction from a dedicated page-initialization seed;
- corresponding prefix page rows therefore start identically across `N`;
- the minibatch RNG is reset after page initialization from a dedicated batch seed;
- every `N` and both routing methods therefore see the same training minibatches for a given replication seed;
- evaluation minibatches are also shared exactly.

### Paired fixed-hash results

| pages | mean validation CE | sample std |
|---:|---:|---:|
| 1 | 2.46687950 | 0.00048935 |
| 4 | 2.46569404 | 0.00032800 |
| 16 | 2.46455893 | 0.00030115 |
| 64 | 2.46409338 | 0.00034050 |
| 256 | **2.46395130** | 0.00037318 |

The fixed-hash curve is strictly monotone in **all three seeds** through 256 pages.

### Paired learned-router results

| pages | mean validation CE | sample std |
|---:|---:|---:|
| 1 | 2.46687950 | 0.00048935 |
| 4 | 2.46534705 | 0.00010996 |
| 16 | 2.46422635 | 0.00021991 |
| 64 | **2.46244999** | 0.00003221 |
| 256 | 2.46245038 | 0.00011370 |

Per-seed `64 -> 256` behavior:

- seed 7: `2.46242460 -> 2.46251403` (regression);
- seed 8: `2.46248622 -> 2.46231911` (improvement);
- seed 9: `2.46243916 -> 2.46251800` (regression).

Thus two of three seeds remain non-monotone at the eight-factor / 256-page step. The three-seed mean change is only about `+0.00000039` CE, i.e. effectively a plateau, but the strict monotonicity criterion still fails.

This replication materially strengthens the original interpretation: the fixed-hash capacity effect is not an artifact of shape-dependent training minibatches, and the learned-router 256-page boundary also survives the stricter pairing.

Successful workflow run: `33327039148`.
Artifact: `g3f-paired-capacity-results`, id `9736565824`, SHA-256 `77b7d9cde86c8fa0e7667a3035c397f18a9ea511dc815cc454646293f1218d41`.

## G3d — conventional flat sparse/MoE control

G3d uses the same frozen backbone and page-shaped `48 -> 10 -> 48` experts, but replaces factorized addressing with a conventional flat resident linear router.

The maximum-width control allocates 256 router outputs for every `N` and always executes the full `48 x 256` projection before slicing active logits:

- flat router parameters: 12,544;
- routing matrix MACs/token: 12,288;
- active expert matrix MACs/token at inference: 960;
- top-2 expert execution during task-only training for `N>1`;
- top-1 expert execution at inference;
- no semantic expert labels or future-token targets;
- no external-I/O claim: experts and flat router are treated as conventional resident MoE state.

Three-seed mean validation CE:

| experts | mean validation CE | sample std |
|---:|---:|---:|
| 1 | 2.46640849 | 0.00082645 |
| 4 | 2.46517187 | 0.00064124 |
| 16 | 2.46378996 | 0.00040788 |
| 64 | **2.46164943** | 0.00036512 |
| 256 | 2.46165924 | 0.00066837 |

The conventional flat MoE is lower-loss than current learned ParamProbe at 64 pages, but it spends about 3.4x as many routing matrix MACs and uses `O(N_max)` resident router state. It also shows a tiny mean `64 -> 256` regression.

This is a deliberately strong quality control, not a valid asymptotic substitute for the ParamProbe resident-memory claim.

Successful workflow run: `33326900388`.
Artifact: `g3de-tinyshakespeare-controls`, id `9736542318`, SHA-256 `9d2f3bbdaa4fd2624d1a5565aafa9e115914d5c5e7ecc3ca22989ccdfbb5abed`.

## G3g — resident dense active-compute baselines

G3g asks whether the learned ParamProbe gain can be reproduced simply by spending similar active dense compute, with no inactive expert/page capacity.

Two resident adapters are trained at the same insertion point with paired minibatch streams:

| dense adapter | matrix MACs/token | mean validation CE | sample std |
|---|---:|---:|---:|
| `48 -> 10 -> 48` | 960 | 2.46710117 | 0.00041528 |
| `48 -> 47 -> 48` | 4,512 | **2.46482733** | 0.00007729 |

The learned ParamProbe maximum-width router plus one active 4 KiB page uses 3,584 + 960 = **4,544 matrix MACs/token**. Its paired 64-page mean is `2.46244999`, about `0.00238` CE below the 4,512-MAC dense control.

Therefore the 64-page result is not explained by merely adding a comparable amount of active dense matrix compute.

Successful workflow run: `33327096295`.
Artifact: `g3g-dense-compute-results`, id `9736558447`, SHA-256 `89e2adf41d943344731c70e9785498473d08469e54f8b38a8c905a19904800f6`.

## G3i — total-compute-matched flat MoE

G3i makes the sparse control much tighter at `N <= 64`.

It uses:

- maximum flat router width: 64 experts;
- flat router: 3,136 resident parameters and 3,072 matrix MACs/token;
- expert: resident `48 -> 15 -> 48`, 1,503 parameters, 6,012 bytes, 1,440 matrix MACs/token;
- total inference matrix work: **4,512 MACs/token**;
- learned ParamProbe total: **4,544 MACs/token**;
- difference: only 32 MACs/token;
- task-only top-2 sparse training and top-1 inference;
- no external-I/O claim.

Three-seed means:

| experts | mean validation CE | sample std |
|---:|---:|---:|
| 1 | 2.46556124 | 0.00036420 |
| 4 | 2.46474781 | 0.00036806 |
| 16 | 2.46347691 | 0.00076380 |
| 64 | **2.46123072** | 0.00015764 |

This flat MoE is about `0.00122` CE lower than paired learned ParamProbe at 64 pages despite essentially identical active matrix compute.

This is an important constraining result. ParamProbe does **not** currently outperform conventional flat sparse routing at finite `N=64`. Its distinguishing property is instead the hard external-parameter probe contract and scalable/factorized resident routing structure. The flat router has resident metadata linear in the maximum expert count and therefore does not satisfy the intended asymptotic resident-memory scaling requirement.

Successful workflow run: `33327309432`.
Artifact: `g3i-compute-matched-moe-results`, id `9736630965`, SHA-256 `f3c9ea7c7c04616e6150fa4780a4c321895cef725a142bb4729dbdd170e58d13`.

## G3h — paired page-granularity sweep

G3h asks a different question from capacity scaling: at fixed **1 MiB physical external capacity**, what page/operator granularity works best under one probe?

The maximum eight-factor fixed hash is shared across all conditions, with 384 routing MACs/token. Each page contains the widest complete FP32 `48 -> h -> 48` residual MLP that fits in the physical block.

| B | pages | hidden width | payload/page | active page MACs/token | mean validation CE | sample std |
|---:|---:|---:|---:|---:|---:|---:|
| 4 KiB | 256 | 10 | 4,072 B | 960 | 2.46392645 | 0.00038834 |
| 16 KiB | 64 | 41 | 16,100 B | 3,936 | **2.46253466** | 0.00023699 |
| 64 KiB | 16 | 168 | 65,376 B | 16,128 | 2.46337525 | 0.00011163 |

The 16 KiB condition is best in **all three seeds**. It wins despite having slightly less learned payload than the other conditions because the physical capacity is exact while complete MLP parameter counts need not perfectly fill every page.

This establishes a non-monotone operator-granularity tradeoff in this diagnostic setup: larger pages provide more local operator capacity but also reduce address count, increase bytes per probe, and increase active compute. More bytes per probe are not automatically better.

The earlier G3e version used unpaired minibatch streams and is retained only as a preliminary run. G3h is the preferred granularity result.

Successful workflow run: `33327170286`.
Artifact: `g3h-paired-granularity-results`, id `9736590384`, SHA-256 `375c62eba62a3f42cd379fddffba429a29162ff0d624fd937db162f242b6b3e1`.

## G3j — exact 4 KiB page-operator-matched flat MoE

This control is running separately to remove the remaining expert-width difference in G3i. It uses a maximum 64-way flat router and the exact `48 -> 10 -> 48`, 4,072-byte expert shape used by ParamProbe. Its total inference matrix work is 3,072 router + 960 expert = 4,032 MACs/token, which is lower than learned ParamProbe's 4,544.

The final table is intentionally not filled until the workflow completes.

## Unit/reference CI

A pinned CPU GitHub Actions workflow now runs the repository test suite on Python 3.12 / PyTorch CPU. The initial CI attempt exposed only an import-path issue for tests that import the repository-local `experiments` namespace. Adding the repository root to `PYTHONPATH` fixed the environment without changing research code.

Successful rerun `33327415104`: **10 tests passed**.

## Interpretation for paper claims

The evidence now separates three questions that were previously entangled.

**Does inactive conditional capacity help under the ParamProbe probe contract?** Yes on this named corpus. The paired fixed-hash curve is strictly monotone through 256 pages with `q=1`, fixed 4 KiB traffic, and fixed active page compute.

**Does the learned factorized router scale cleanly through 256 pages?** Not yet. Pairing makes the three-seed mean essentially flat at 64 versus 256, and two seeds still regress. The eight-factor address-reliability edge remains a genuine constraint.

**Is the current learned ParamProbe router better than a conventional finite flat MoE?** No. A compute-matched task-trained flat MoE is lower-loss at 64 experts. The ParamProbe contribution must therefore be argued from the resource contract and scalable routing/storage structure, not from finite-`N` routing quality superiority.

The next publication-strengthening work should target generalization across a second named corpus and then physical/file-backed execution of trained LM pages, while keeping the Tiny Shakespeare tables frozen.