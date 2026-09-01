# G7b: balanced q=1 product-key challenge result

Status: **FINITE-N UNIQUENESS KILLED**

G7b was predeclared after G7 exposed severe utilization collapse in the naive top-1 product-key adaptation. G7b changes only training-time router regularization while preserving the same hard inference resource envelope and uses fresh seeds 35/36/37.

## Frozen hard inference contract

Both methods use:

- N=256 external pages;
- page operator 96 -> 20 -> 96;
- 3,956 FP32 parameters/page = 15,824 learned bytes;
- physical block B=16,384 bytes;
- q=1 selected page/token;
- exactly 16,384 logical external parameter bytes/token;
- 3,840 active page matrix MACs/token;
- the same frozen G6a WikiText-2 subword backbone/tokenizer;
- 1,920 page-training steps with mean 7,680 routed page assignments/page.

Fixed factorized router:

- 3,104 bytes fixed metadata;
- 768 matrix MACs/token.

Balanced PEER-style product-key router (d_key=6):

- 2,712 learned parameter bytes + 56 bytes BatchNorm running-state buffers;
- 672 router matrix/dot MACs/token;
- no increase in q, B, page operator, or selected page compute.

## Predeclared utilization gate

Before CE comparison, the product-key router had to satisfy in every seed:

- validation normalized utilization entropy >= 0.85;
- dead-page fraction <= 0.05.

All three seeds passed.

| seed | PK util entropy | PK dead-page fraction | utilization gate |
|---:|---:|---:|:---:|
| 35 | 0.92099580 | 0.00000000 | pass |
| 36 | 0.94724534 | 0.00000000 | pass |
| 37 | 0.96187583 | 0.03906250 | pass |

The training-only router pretraining also produced high route entropy before page training (0.9214 / 0.9470 / 0.9619), so the G7 collapse confound was successfully removed.

## Exact CE results

| seed | fixed factorized CE | balanced product-key CE | product-key better |
|---:|---:|---:|:---:|
| 35 | 5.69951464 | **5.69847822** | yes |
| 36 | 5.69932947 | **5.69776409** | yes |
| 37 | 5.69947526 | **5.69833002** | yes |

Means +/- sample standard deviation:

- fixed factorized: `5.69943979 +/- 0.00009755`;
- balanced product-key: `5.69819078 +/- 0.00037688`;
- mean delta product-key - fixed: **`-0.00124902` CE**.

Balanced product-key wins all three fresh seeds while also using fewer finite-N router MACs and fewer router parameter/metadata bytes.

Route stability under the predeclared perturbation diagnostic is also higher for balanced product-key:

- fixed: about 0.7698 mean;
- balanced product-key: about 0.8283 mean.

## Predeclared classification

The G7b predeclared classification is therefore:

`g7b_classification=finite_n_uniqueness_killed`

This kills any claim that the current fixed factorized ParamProbe router occupies a uniquely superior finite-N q=1 quality/resource point at N=256.

## What survives

This result does **not** kill the parameter-probe research program. It narrows the contribution.

What remains supported:

1. The explicit hard external parameter-block probe formalism and exact qB accounting.
2. Complete nonlinear micro-operators can be serialized and executed through a one-block file-backed path.
3. External inactive capacity can improve LM quality under a fixed inference probe/page-compute envelope.
4. Product-key routing can satisfy the same q=1 block contract at N=256 and outperform the current factorized router when trained to avoid collapse.
5. The surviving router distinction is asymptotic/resource scaling: the current binary factorized address family requires routing work/metadata proportional to address width Theta(log N), while standard two-bank product-key routing requires Theta(sqrt(N) * d_key) key scoring/state for fixed key dimension.

Accordingly, the paper must not claim finite-N routing superiority over product-key methods. Any architecture-specific advantage now has to be demonstrated at capacities where the different address-scaling laws materially separate, or the contribution should be framed primarily as a parameter-I/O resource model and systems study.

## Reproducibility

- workflow run: `33361911914`;
- workflow head SHA: `fb6a91c866469d62be83322ac890e30bada94f80`;
- artifact: `g7b-balanced-product-key-results`;
- artifact id: `9747159785`;
- artifact SHA-256: `5a88157971d95bfd12d2273d0954694f19ca4c93f5ceeb40da15ecbf49437da2`.

No additional product-key hyperparameter tuning is authorized in this experiment family after this result.