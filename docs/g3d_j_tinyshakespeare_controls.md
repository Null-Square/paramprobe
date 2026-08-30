# G3d–G3j — Tiny Shakespeare controls and robustness

## Status

**FROZEN TINY SHAKESPEARE CONTROL SUITE — MIXED / CONSTRAINING, WITH A ROBUST CAPACITY EFFECT.**

This suite freezes the original G3c Tiny Shakespeare result and tests the main reviewer-facing alternatives without tuning the learned-router boundary.

The consolidated conclusions are:

1. the fixed-hash external-capacity curve remains strictly monotone through 256 pages under a stricter paired training protocol;
2. the learned causal router again improves through 64 pages and is effectively flat / slightly non-monotone at 256 pages, so the eight-factor address-reliability boundary survives pairing;
3. a resident dense adapter matched to learned ParamProbe's total active matrix compute does not match its 64-page validation loss;
4. a conventional task-trained flat MoE beats the current learned ParamProbe router at 64 experts, even with the exact same 4 KiB expert shape and less active matrix compute, but uses flat `O(N)` resident routing state and makes no hard external-I/O claim;
5. at fixed 1 MiB physical external capacity, a paired page-granularity sweep favors 16 KiB pages over both 4 KiB and 64 KiB pages in all three seeds;
6. the ordinary repository test suite passes 10/10 on pinned CPU CI.

These results support the usefulness of inactive conditional capacity under a hard probe budget. They do **not** support a claim that the present ParamProbe learned router is better than conventional finite flat MoE routing.

## Common dataset and backbone

All runs use the canonical Karpathy `char-rnn` Tiny Shakespeare file:

- bytes: `1,115,394`;
- SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`;
- Git blob SHA-1: `7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9`;
- first 90% train / final 10% validation;
- byte-level 256-symbol modeling;
- frozen two-block Transformer backbone validation CE: `2.47749685`.

The successful insertion is

`embedding -> Transformer block 1 -> conditional operator -> Transformer block 2 -> LM head`.

The standard ParamProbe page is `48 -> 10 -> 48`: 1,018 FP32 parameters, 4,072 learned bytes, one 4,096-byte physical page, and 960 active page matrix MACs/token. The inference contract is `q=1`, so logical external parameter traffic is exactly 4,096 bytes/token in the 4 KiB capacity sweep.

## G3f — paired capacity replication

G3f removes a nuisance variable in G3c: differently sized page tables consumed different amounts of RNG during initialization before minibatches were sampled. G3f explicitly reinitializes page prefixes from a dedicated seed and resets the minibatch RNG after page construction. Thus all `N` values and both routing methods see matched training and evaluation examples for a given seed.

### Fixed-hash results

| pages | mean validation CE | sample std |
|---:|---:|---:|
| 1 | 2.46687950 | 0.00048935 |
| 4 | 2.46569404 | 0.00032800 |
| 16 | 2.46455893 | 0.00030115 |
| 64 | 2.46409338 | 0.00034050 |
| 256 | **2.46395130** | 0.00037318 |

The fixed-hash curve is strictly monotone in **all three seeds** through 256 pages.

### Learned-router results

| pages | mean validation CE | sample std |
|---:|---:|---:|
| 1 | 2.46687950 | 0.00048935 |
| 4 | 2.46534705 | 0.00010996 |
| 16 | 2.46422635 | 0.00021991 |
| 64 | **2.46244999** | 0.00003221 |
| 256 | 2.46245038 | 0.00011370 |

At `64 -> 256`, seeds 7 and 9 regress while seed 8 improves. The three-seed mean changes by only about `+0.00000039` CE, effectively a plateau, but the predeclared strict monotonicity criterion still fails.

This paired replication therefore strengthens both sides of G3c: the fixed-router capacity effect survives stricter pairing, and the learned eight-factor boundary also survives.

Workflow run: `33327039148`. Artifact `g3f-paired-capacity-results`, id `9736565824`, SHA-256 `77b7d9cde86c8fa0e7667a3035c397f18a9ea511dc815cc454646293f1218d41`.

## G3g — resident dense compute controls

These controls ask whether the 64-page gain can be reproduced merely by spending comparable active dense compute with no inactive external capacity.

| resident adapter | matrix MACs/token | mean validation CE | sample std |
|---|---:|---:|---:|
| `48 -> 10 -> 48` | 960 | 2.46710117 | 0.00041528 |
| `48 -> 47 -> 48` | 4,512 | **2.46482733** | 0.00007729 |

Learned ParamProbe uses 3,584 router + 960 active-page = **4,544 matrix MACs/token** and reaches `2.46244999` at 64 pages in the paired run, about `0.00238` CE below the 4,512-MAC dense adapter.

Thus the 64-page benefit is not explained by simply adding a comparable amount of active dense matrix compute.

Workflow run: `33327096295`. Artifact `g3g-dense-compute-results`, id `9736558447`, SHA-256 `89e2adf41d943344731c70e9785498473d08469e54f8b38a8c905a19904800f6`.

## G3d — maximum-width flat MoE quality control

A conventional resident flat router with maximum width 256 executes a full `48 x 256` projection for every condition and uses the same `48 -> 10 -> 48` expert shape.

- flat router: 12,544 parameters, 12,288 matrix MACs/token;
- active expert: 960 matrix MACs/token;
- task-only top-2 training and top-1 inference;
- no semantic addresses or future-token routing targets;
- no external-I/O claim.

Three-seed means are `2.46640849`, `2.46517187`, `2.46378996`, `2.46164943`, and `2.46165924` for 1/4/16/64/256 experts respectively. It improves strongly through 64 and has a tiny mean 64→256 regression.

This is a deliberately strong finite-`N` quality control, but its router state and compute scale linearly with the maximum expert count.

Workflow run: `33326900388`. Artifact `g3de-tinyshakespeare-controls`, id `9736542318`, SHA-256 `9d2f3bbdaa4fd2624d1a5565aafa9e115914d5c5e7ecc3ca22989ccdfbb5abed`.

## G3i — total-compute-matched flat MoE

G3i tightens the sparse comparison to maximum `N=64`:

- flat router: 64 outputs, 3,136 parameters, 3,072 matrix MACs/token;
- resident expert: `48 -> 15 -> 48`, 1,503 parameters, 1,440 matrix MACs/token;
- total inference matrix work: **4,512 MACs/token**;
- learned ParamProbe total: **4,544 MACs/token**.

| experts | mean validation CE | sample std |
|---:|---:|---:|
| 1 | 2.46556124 | 0.00036420 |
| 4 | 2.46474781 | 0.00036806 |
| 16 | 2.46347691 | 0.00076380 |
| 64 | **2.46123072** | 0.00015764 |

At essentially identical active matrix compute, this flat MoE is about `0.00122` CE lower than paired learned ParamProbe at 64 pages.

Workflow run: `33327309432`. Artifact `g3i-compute-matched-moe-results`, id `9736630965`, SHA-256 `f3c9ea7c7c04616e6150fa4780a4c321895cef725a142bb4729dbdd170e58d13`.

## G3j — exact 4 KiB page-operator-matched flat MoE

G3j removes the remaining expert-width difference from G3i. It uses the exact same 4 KiB page-shaped expert as ParamProbe:

- flat router: maximum 64 outputs, 3,136 parameters, 3,072 matrix MACs/token;
- expert: exact `48 -> 10 -> 48`, 1,018 parameters, 4,072 learned bytes, 960 matrix MACs/token;
- total inference matrix work: **4,032 MACs/token**;
- learned ParamProbe total: **4,544 MACs/token**;
- flat MoE therefore uses **512 fewer** active matrix MACs/token.

| experts | mean validation CE | sample std |
|---:|---:|---:|
| 1 | 2.46640849 | 0.00082645 |
| 4 | 2.46516396 | 0.00063630 |
| 16 | 2.46373103 | 0.00052195 |
| 64 | **2.46182543** | 0.00038872 |

The exact page-shape-matched flat MoE still beats paired learned ParamProbe at 64 pages by about `0.00062456` CE while using less active matrix compute. This confirms that G3i's advantage is not caused merely by its wider expert.

This is the strongest current finite-`N` routing limitation: the present factorized learned router is not quality-superior to a conventional 64-way flat router. The distinction is instead the resource structure. Flat routing carries resident metadata linear in `N` and these controls keep experts resident; they do not demonstrate a bounded-resident-memory, hard external-parameter-probe implementation.

Workflow run: `33327465444`. Artifact `g3j-page-matched-moe-results`, id `9736674274`, SHA-256 `7e219908484f990c0b06bff290cb01f87b242a51770959bac25758973ef28e1f`.

## G3h — paired page granularity

At fixed **1 MiB physical external capacity**, the maximum eight-factor fixed hash is shared across all conditions and each page contains the widest complete FP32 `48 -> h -> 48` MLP that fits.

| B | pages | hidden width | active page MACs/token | mean validation CE | sample std |
|---:|---:|---:|---:|---:|---:|
| 4 KiB | 256 | 10 | 960 | 2.46392645 | 0.00038834 |
| 16 KiB | 64 | 41 | 3,936 | **2.46253466** | 0.00023699 |
| 64 KiB | 16 | 168 | 16,128 | 2.46337525 | 0.00011163 |

The 16 KiB condition is best in **all three seeds**. Larger pages trade fewer addressable operators for more local operator capacity, bytes/probe, and active compute; more bytes per probe are therefore not automatically better.

Workflow run: `33327170286`. Artifact `g3h-paired-granularity-results`, id `9736590384`, SHA-256 `375c62eba62a3f42cd379fddffba429a29162ff0d624fd937db162f242b6b3e1`.

## Unit/reference CI

Pinned CPU CI now runs the repository tests under Python 3.12 / PyTorch 2.10 CPU. The first attempt exposed only an import-path setup issue for tests importing the repository-local `experiments` namespace; adding the repository root to `PYTHONPATH` fixed the environment without changing research code.

Successful rerun `33327415104`: **10 tests passed**.

## Interpretation for paper claims

The controls separate three claims that must not be conflated.

**Inactive conditional capacity helps under the hard probe contract.** Supported on Tiny Shakespeare: the paired fixed-hash curve is strictly monotone through 256 pages with `q=1`, fixed 4 KiB logical external traffic, and fixed active page compute.

**The current learned factorized router scales cleanly through 256 pages.** Not supported. It is strong through 64 pages and effectively plateaus at 256, with two of three paired seeds regressing.

**The current learned ParamProbe router beats a conventional finite flat MoE.** Not supported. The exact page-shape-matched 64-way flat MoE is lower-loss while using fewer active matrix MACs. ParamProbe's empirical contribution must therefore be framed around the hard external-parameter probe contract, bounded/scalable routing structure, and capacity scaling—not finite-`N` routing-quality superiority.

The next publication-strengthening work should freeze these Tiny Shakespeare results, replicate the capacity effect on a second named corpus, and validate trained LM pages through serialized/file-backed external execution before making a Q1-ready claim.