# G6d: predeclared larger-LM capacity sweep with explicit training-budget scaling

Status before execution: **PREDECLARED / UNRUN**

## Purpose

G6a is frozen as a failure under a fixed total page-training budget: `N=16` improves validation CE, while `N=256` regresses in all three seeds.

G6c independently confirms that sparse page training exposure is a major cause: with fresh seeds, `N=256` is worse than `N=16` at 120 steps but crosses below the paired `N=16` reference once both capacities receive the same mean routed assignments/page.

G6d asks the next, distinct question:

> Does the larger subword-LM recover a capacity trend across `N=1,4,16,64,256` when page-training budget is an explicit measured resource that scales only where needed to maintain a minimum mean routed exposure/page?

This does **not** retry or redefine G6a. G6a remains the fixed-training-budget result.

## Frozen assets

Use the exact archived G6a assets from workflow run `33333431198`, artifact `9738331594`:

- tokenizer SHA-256: `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`;
- frozen backbone SHA-256: `a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049`.

Do not retrain either asset.

Dataset remains the verified WikiText-2 raw train/validation split, archive SHA-256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.

## Frozen inference configuration

Identical to G6a/G6c:

- train-only byte-level BPE vocabulary 1,024;
- causal Transformer: `d_model=96`, 3 layers, 4 heads, FF 384, context 128;
- insertion after block 2;
- fixed balanced maximum-width 8-factor hash;
- projection seed 999 and train-only median thresholds from 40 calibration batches x 8;
- maximum routing width fixed at eight bits for every capacity;
- fixed router matrix MACs/token: 768 for every capacity;
- page operator `96 -> 20 -> 96`, tanh, residual scale 0.15;
- 3,956 FP32 parameters/page = 15,824 learned bytes;
- physical block size `B=16,384` bytes;
- active page matrix MACs/token: 3,840;
- `q=1` and exactly 16,384 logical external parameter bytes/token.

Inference router/page compute and explicit external parameter traffic are fixed across `N`.

## Capacities and fresh page seeds

Capacity sweep: `N=1,4,16,64,256`, corresponding to used address bits `0,2,4,6,8`.

Fresh page-training seeds: **13, 14, 15**, chosen before execution.

For each seed:

- page initialization seed: `40000 + seed`;
- hidden-state page-training bank seed: `50000 + seed`;
- every capacity uses a prefix of the same frozen 1,920-minibatch hidden/target bank;
- page initialization is paired by reusing the same explicit initialization seed after table construction, so corresponding prefix rows share initialization across capacities.

## Predeclared training-budget rule

Page optimizer remains AdamW, lr `4e-3`, weight decay `1e-4`.

Each minibatch contains `8 x 128 = 1,024` routed token assignments.

Define the minimum target mean routed exposure/page as **7,680 assignments/page**, matching the successful G6c equal-exposure crossover.

Use the deterministic rule

`steps(N) = max(120, ceil(7680 * N / 1024))`.

For the predeclared capacities this is exactly:

| pages | page-training steps | total routed assignments | mean assignments/page |
|---:|---:|---:|---:|
| 1 | 120 | 122,880 | 122,880 |
| 4 | 120 | 122,880 | 30,720 |
| 16 | 120 | 122,880 | 7,680 |
| 64 | 480 | 491,520 | 7,680 |
| 256 | 1,920 | 1,966,080 | 7,680 |

Thus G6d preserves the original 120-step G6a schedule through `N=16` and increases total page-training work only for larger tables whose mean routed exposure/page would otherwise fall below the G6c-supported floor.

This is **not** a fixed-total-training-compute experiment. Total training resource is reported explicitly and grows with `N` beyond 16 pages.

## Evaluation

Reuse the frozen G6a evaluation protocol:

- official WikiText-2 raw validation split;
- RNG seed 1234;
- 40 batches x 8 sequences;
- context 128;
- identical frozen evaluation hidden-state bank for every capacity and seed.

Report per-seed CE, three-seed mean/sample std, route-utilization entropy, dead-page fraction, page-training steps, total routed assignments, and mean assignments/page.

## Primary criterion

G6d supports a **larger-model capacity trend under explicit capacity-scaled training resources** only if all of the following hold:

1. the three-seed mean validation CE is strictly decreasing across `N=1,4,16,64,256`;
2. `N=256` beats `N=1` in every fresh seed 13/14/15;
3. `N=256` beats `N=16` in every fresh seed 13/14/15;
4. the frozen inference-resource assertions hold exactly;
5. the predeclared training-budget rule is followed exactly.

Full per-seed adjacent-capacity monotonicity is reported as a secondary robustness diagnostic but is **not** required for this gate, consistent with the project's retained evidence that small local reversals can occur even when the capacity trend is robust.

No validation result may be used to change the training-budget rule or hyperparameters.

## Interpretation boundary

If G6d passes, the supported claim is narrow:

> At this larger subword-LM scale, increasing inactive external page capacity can improve validation loss while inference traffic and active page compute remain fixed, provided page-training optimization/sample exposure is allowed to scale as an explicit resource with external capacity.

This would not establish capacity gains at fixed total training compute and would not imply an optimal `Theta(N)` training law. It would establish that the G6a failure is not a fundamental inability of the fixed factorized ParamProbe architecture to use 256 pages at this model scale.

If G6d fails, preserve the result and analyze whether the remaining limitation is capacity-dependent optimization, routing partition quality, operator size, or insertion depth rather than changing the frozen G6d protocol.
