# G6d: larger subword-LM capacity sweep with explicit training-budget scaling

Status: **PRIMARY GATE FAILED; LARGE-CAPACITY ENDPOINT BENEFIT RETAINED**

G6d was predeclared before implementation/execution in `docs/g6d_capacity_scaled_training_predeclared.md`. It uses the exact archived G6a tokenizer/backbone, the same fixed factorized router and 16 KiB page operator, fresh page seeds 13/14/15, and an explicit capacity-dependent page-training budget treated as a measured resource.

## Frozen inference resources

- tokenizer SHA-256: `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`
- backbone SHA-256: `a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049`
- WikiText-2 raw archive SHA-256: `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`
- `q=1`
- `B=16,384` bytes
- logical external parameter traffic: 16,384 bytes/token
- page operator: `96 -> 20 -> 96`
- page payload: 15,824 learned bytes in one block
- active page matrix MACs/token: 3,840
- fixed-router matrix MACs/token: 768
- maximum address width: eight bits, allocated once for every capacity

## Predeclared training-resource rule

`steps(N) = max(120, ceil(7680 * N / 1024))`

| pages | steps | total routed assignments | mean assignments/page | utilization entropy | dead-page fraction |
|---:|---:|---:|---:|---:|---:|
| 1 | 120 | 122,880 | 122,880 | 1.00000000 | 0 |
| 4 | 120 | 122,880 | 30,720 | 0.99586189 | 0 |
| 16 | 120 | 122,880 | 7,680 | 0.92041132 | 0 |
| 64 | 480 | 491,520 | 7,680 | 0.90679623 | 0 |
| 256 | 1,920 | 1,966,080 | 7,680 | 0.90570751 | 0 |

Inference resources stay fixed; total page-training work grows beyond 16 pages.

## Fresh-seed results

| seed | N=1 | N=4 | N=16 | N=64 | N=256 |
|---:|---:|---:|---:|---:|---:|
| 13 | 5.70271982 | 5.70230366 | 5.70234591 | 5.70093304 | 5.69960475 |
| 14 | 5.70267535 | 5.70240639 | 5.70251317 | 5.70110936 | 5.69940451 |
| 15 | 5.70267190 | 5.70247083 | 5.70245014 | 5.70106655 | 5.69944949 |

Three-seed mean +/- sample std:

| pages | steps | validation CE |
|---:|---:|---:|
| 1 | 120 | 5.70268902 +/- 0.00002673 |
| 4 | 120 | 5.70239363 +/- 0.00008431 |
| 16 | 120 | 5.70243641 +/- 0.00008447 |
| 64 | 480 | 5.70103632 +/- 0.00009197 |
| 256 | 1,920 | 5.69948625 +/- 0.00010506 |

Mean adjacent changes are approximately:

- `1 -> 4`: `-0.00029539`
- `4 -> 16`: `+0.00004278`
- `16 -> 64`: `-0.00140009`
- `64 -> 256`: `-0.00155007`

The predeclared primary gate required a strictly decreasing three-seed mean across all five capacities. It therefore **fails** because the mean has a small `4 -> 16` reversal.

Other predeclared outcomes:

- `N=256` beats `N=1` in 3/3 fresh seeds;
- `N=256` beats `N=16` in 3/3 fresh seeds;
- full per-seed monotonicity holds in seed 15 only; seeds 13 and 14 contain the same small `4 -> 16` reversal direction;
- resource gate passes;
- training-budget-rule gate passes.

Therefore `g6d_capacity_trend_passed=false`.

## Interpretation

G6d does **not** justify a strict monotone larger-model capacity curve under the declared training rule. That negative result is frozen.

At the same time, it materially strengthens the narrower large-capacity result: once the G6c-supported exposure floor is supplied, 64 and 256 pages substantially outperform the 16-page condition, and 256 pages beats both 1 and 16 pages in every fresh seed. The only failed edge is the much smaller `4 -> 16` difference under the unchanged 120-step budget.

The next scientifically clean step is not to tune G6d. It is a separately predeclared paired `N=4` versus `N=16` robustness/estimation study with many fresh seeds under the exact same 120-step protocol. That study can estimate whether the local reversal is a stable low-capacity effect or ordinary page-training stochasticity. It cannot retroactively change the G6d gate outcome.

## Reproducibility

- workflow run: `33334607629`
- artifact: `g6d-capacity-scaled-training-results`
- artifact id: `9738729212`
- artifact SHA-256: `a140566290b81db03503ef9a6e6ca0ba53c70e316588e58fcb6106ce62620690`
- workflow head SHA: `50e4c973de2a26e615c21616bd8afe1fc8f5e25f`
