# G6c: predeclared paired training-exposure crossover

Status before execution: **PREDECLARED / UNRUN**

## Purpose

G6a is a frozen scientific failure: with 120 page-training steps, `N=16` improves validation CE but `N=256` regresses in all three seeds.

G6b attempted to test sparse training exposure but failed an over-strict cross-run absolute replication tolerance; its apparently positive trajectory is explicitly not counted as scientific evidence.

G6c is an independent confirmatory test that removes the cross-run absolute-CE dependency. It uses fresh page seeds and compares `N=16` and `N=256` **inside the same workflow/run** with a paired training-data prefix.

## Frozen assets

Use the exact archived G6a assets from workflow run `33333431198`, artifact `9738331594`:

- tokenizer SHA-256: `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`;
- frozen backbone SHA-256: `a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049`.

Do not retrain either asset.

Dataset remains the verified WikiText-2 raw official train/validation split, archive SHA-256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.

## Frozen model/router/page configuration

Identical to G6a:

- train-only byte-level BPE vocabulary 1,024;
- causal Transformer: `d_model=96`, 3 layers, 4 heads, FF 384, context 128;
- insertion after block 2;
- fixed balanced maximum-width 8-factor hash;
- projection seed 999 and train-only median thresholds from 40 calibration batches x 8;
- fixed router matrix MACs/token: 768;
- page operator `96 -> 20 -> 96`, tanh, residual scale 0.15;
- 3,956 FP32 parameters/page = 15,824 learned bytes;
- physical block size `B=16,384` bytes;
- active page matrix MACs/token: 3,840;
- `q=1` and exactly 16,384 logical external parameter bytes/token;
- page optimizer AdamW, lr `4e-3`, weight decay `1e-4`.

No learned router or operator/page-size change is allowed.

## Fresh page seeds

Use page-training seeds **10, 11, 12**, chosen before G6c execution.

For each seed:

- page initialization seed: `40000 + seed`;
- page minibatch seed: `50000 + seed`;
- materialize one frozen hidden-state training bank of 1,920 minibatches using the exact G6a `make_hidden_bank` implementation;
- both capacity conditions use the same first 120 hidden-state/target minibatches;
- page initialization is paired by using the same initialization seed for `N=16` and `N=256`.

## Paired training conditions

### N=16 reference

Train 16 pages for exactly 120 steps on the first 120 minibatches, then evaluate.

Total routed token assignments: `120 * 8 * 128 = 122,880`.

Mean assignments/page: `122,880 / 16 = 7,680`.

### N=256 exposure trajectory

Train 256 pages continuously for 1,920 steps on the full frozen bank.

Evaluate at cumulative checkpoints:

`120, 240, 480, 960, 1920`.

Mean assignments/page at those checkpoints are:

`480, 960, 1,920, 3,840, 7,680`.

Thus `N=256 @ 1920` and `N=16 @ 120` have equal **mean routed training assignments/page** while the 256-page system uses 16x more total page-training token assignments/optimization compute.

This is a training-resource diagnostic, not a matched-total-training-compute comparison.

## Evaluation

Reuse the frozen G6a evaluation protocol within the same run:

- official validation split;
- RNG seed 1234;
- 40 batches x 8 sequences;
- context 128;
- identical evaluation hidden-state bank for both capacities and all checkpoints.

Report CE, route-utilization entropy, and dead-page fraction.

All research shell pipelines must propagate Python failures (`set -o pipefail`).

## Primary confirmatory criterion

The **training-exposure crossover is confirmed** only if every fresh seed `10/11/12` satisfies all of:

1. **low-exposure replication direction:** `CE(N=16, 120) < CE(N=256, 120)`;
2. **equal-per-page-exposure crossover:** `CE(N=256, 1920) < CE(N=16, 120)`;
3. **within-N improvement:** `CE(N=256, 1920) < CE(N=256, 120)`;
4. frozen inference-resource assertions hold exactly.

Checkpoint-to-checkpoint monotonicity of N=256 is reported but is not required.

No absolute CE equality to an earlier hosted run is required; all decisive comparisons are within-run and paired.

## Interpretation boundary

If confirmed, G6c supports the statement that **training exposure/optimization is a major cause of the G6a 256-page failure**. It still does not make G6a pass and does not show capacity scaling at fixed training cost.

A confirmed crossover would justify a separately predeclared G6d capacity sweep with an explicit training-budget scaling rule treated as a measured resource.

If G6c fails, the exposure explanation is not confirmed and subsequent work should target scale-dependent routing/operator/insertion mismatches rather than simply increasing training duration.
