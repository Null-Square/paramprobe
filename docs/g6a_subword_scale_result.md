# G6a: larger subword-LM fixed-router capacity gate

Status: **FAILED — 16 PAGES IMPROVE, 256 PAGES REGRESS IN ALL THREE SEEDS**

G6a was predeclared in `docs/g6a_subword_scale_predeclared.md` before the experiment code/run. The result below is frozen. No G6a hyperparameter is retuned after observing it.

## What changed from the tiny byte-LM studies

G6a intentionally changes model/tokenization scale rather than router design:

- WikiText-2 raw official train/validation split;
- train-only byte-level BPE, vocabulary 1,024;
- 446,304-parameter causal Transformer backbone;
- `d_model=96`, 3 layers, 4 heads, FF width 384;
- context 128 subword tokens;
- ParamProbe insertion after block 2, before block 3;
- fixed balanced 8-factor hash only;
- page block size `B=16,384` bytes;
- page operator `96 -> 20 -> 96`;
- 3,956 FP32 parameters/page = 15,824 learned bytes padded to one 16 KiB block;
- active page matrix MACs/token = 3,840;
- fixed router matrix MACs/token = 768;
- `q=1`, exactly 16,384 logical external parameter bytes/token;
- capacity points `N=1,16,256`;
- paired page initialization and paired page-training minibatches;
- page seeds `7,8,9`.

Tokenizer fitting uses only `wiki.train.raw`. The tokenizer JSON is 53,272 bytes with SHA-256 `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`.

Tokenized split sizes:

- train: 4,254,523 tokens;
- validation: 445,470 tokens.

Frozen-backbone validation CE: `5.70919060`.

## Exact results

| seed | 1 page | 16 pages | 256 pages | strict `1>16>256` |
|---:|---:|---:|---:|:---:|
| 7 | 5.70266097 | 5.70245363 | 5.70341599 | no |
| 8 | 5.70267036 | 5.70234185 | 5.70355248 | no |
| 9 | 5.70258125 | 5.70222968 | 5.70339634 | no |

Mean +/- sample standard deviation:

| pages | validation CE | utilization entropy | dead-page fraction |
|---:|---:|---:|---:|
| 1 | `5.70263753 +/- 0.00004896` | 1.00000000 | 0.00000000 |
| 16 | `5.70234172 +/- 0.00011197` | 0.92041132 | 0.00000000 |
| 256 | `5.70345494 +/- 0.00008505` | 0.90570751 | 0.00000000 |

Paired mean CE changes:

- `1 -> 16`: `-0.00029581`;
- `16 -> 256`: `+0.00111322`;
- `1 -> 256`: `+0.00081741`.

Every seed improves at 16 pages and every seed regresses at 256 pages. The predeclared G6a gate therefore fails decisively.

## Resource and leakage checks

The executable assertions passed:

- page payload exactly 15,824 bytes;
- physical block exactly 16,384 bytes;
- one selected page/token (`q=1`);
- exactly 16,384 logical external parameter bytes/token;
- page active matrix compute fixed at 3,840 MACs/token;
- maximum router width fixed at 8 factors for all capacities;
- fixed router matrix compute fixed at 768 MACs/token;
- tokenizer fitting and hash threshold calibration use training data only;
- all 256 pages receive validation traffic (`dead_page_fraction=0`).

Thus the failure is not explained by a violated inference resource envelope, validation leakage, or trivially unused pages.

## Training-exposure diagnosis

G6a intentionally kept the page-training schedule fixed at 120 minibatches of 8 x 128 tokens for every capacity. That produces 122,880 routed token assignments per page-training run.

Ignoring routing imbalance, mean training assignments/page are therefore approximately:

- `N=1`: 122,880;
- `N=16`: 7,680;
- `N=256`: 480.

The 256-page route is not sparse in validation usage—normalized entropy is about 0.906 and no page is dead—but each page receives roughly sixteen times fewer training assignments than at `N=16` under the fixed-step schedule.

This makes sparse page sample exposure / optimization a concrete post-result hypothesis. It is **not established causally** by G6a, and the failed gate is not reinterpreted as a pass.

## Interpretation

G6a falsifies the simplest transfer of the tiny-LM result:

> Merely increasing addressable inactive page capacity does not guarantee a benefit at larger model/tokenization scale under a fixed page-training budget.

The result suggests that training sample complexity may itself become a scaling resource even though inference traffic and active inference compute remain bounded.

That distinction matters for the paper: the architectural hard-probe claim survives, but empirical capacity scaling may require a training procedure whose optimization/sample budget grows appropriately with external capacity.

## Reproducibility

- predeclaration commit: `4e9459e2a755bab7e1c6228bf252807f7303f4f6`;
- experiment code commit: `42040a14c50a5f8a29dbb6e90c9703652a346390`;
- workflow commit: `9f9f7afd4b6cf6e3ceddcb57a726ca815293c3f6`;
- workflow run: `33333431198`;
- artifact id: `9738331594`;
- artifact SHA-256: `0e901efe5d360b905fbb15c5cdb5d90ffea3ad60d7f05e2712fe43fc37d2ad21`.

The artifact includes the result log, dataset identity records, tokenizer JSON, frozen backbone checkpoint, model/tokenizer hashes, and environment record.
