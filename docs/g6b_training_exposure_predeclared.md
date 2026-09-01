# G6b: predeclared training-exposure diagnostic

Status before execution: **PREDECLARED / UNRUN**

## Motivation

Frozen G6a fails at larger capacity under a fixed page-training schedule:

| seed | G6a N=16 @ 120 steps | G6a N=256 @ 120 steps |
|---:|---:|---:|
| 7 | 5.70245363 | 5.70341599 |
| 8 | 5.70234185 | 5.70355248 |
| 9 | 5.70222968 | 5.70339634 |

G6a gives every capacity the same 120 minibatches of `8 x 128` routed tokens = 122,880 routed token assignments. Mean assignments/page are therefore approximately 7,680 at `N=16` but only 480 at `N=256`.

G6b tests the post-G6a hypothesis that this sparse training exposure is a major cause of the 256-page regression.

G6b is **not** a retry or redefinition of G6a. G6a remains failed. G6b changes training duration only and reports that extra training compute explicitly.

## Frozen assets

Use the exact G6a workflow artifact from run `33333431198`, artifact `9738331594`:

- tokenizer JSON SHA-256: `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`;
- frozen backbone checkpoint SHA-256: `a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049`.

The workflow must download and verify these exact artifacts. It must not retrain the tokenizer or backbone.

Dataset remains the same verified WikiText-2 raw official train/validation split, with archive SHA-256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.

## Frozen architecture and inference resources

Everything except page-training duration remains identical to G6a:

- byte-level BPE vocabulary 1,024;
- causal Transformer: `d_model=96`, 3 layers, 4 heads, FF 384, context 128;
- insertion after block 2;
- fixed balanced maximum-width 8-factor hash;
- projection seed 999, train-only median thresholds, 40 calibration batches x 8;
- fixed router matrix MACs/token: 768;
- page operator: `96 -> 20 -> 96`, tanh, residual scale 0.15;
- 3,956 FP32 parameters/page = 15,824 learned bytes;
- physical block size `B=16,384` bytes;
- active page matrix MACs/token: 3,840;
- `q=1`;
- exactly 16,384 logical external parameter bytes/token;
- capacity: **N=256 only**;
- page seeds: 7, 8, 9;
- page initialization seed base: `40000 + seed`;
- page minibatch seed base: `50000 + seed`;
- AdamW page optimizer: lr `4e-3`, weight decay `1e-4`.

No learned router is introduced.

## Continuous training trajectory

For each page seed, construct the same N=256 page table and optimizer as G6a, set the same page-minibatch RNG seed, and train one **continuous** trajectory for 1,920 steps.

Evaluate checkpoints after cumulative steps:

`120, 240, 480, 960, 1920`.

Do not restart the optimizer between checkpoints.

The first 120 training minibatches must be the exact deterministic prefix used in G6a. As an integrity check, G6b's 120-step CE must reproduce the frozen G6a N=256 value for each seed within absolute tolerance `2e-7`. Failure of this replication assertion is infrastructure/protocol failure, not scientific evidence.

Mean routed assignments/page at each checkpoint, assuming balance, are:

- 120 steps: 480/page;
- 240: 960/page;
- 480: 1,920/page;
- 960: 3,840/page;
- 1,920: 7,680/page.

Thus `N=256 @ 1920` matches the **mean training assignments/page** of frozen `N=16 @ 120` while using 16x more total page-training token assignments and training compute.

## Evaluation

Reuse the exact G6a evaluation protocol:

- official WikiText validation split;
- evaluation RNG seed 1234;
- 40 batches x 8 sequences;
- context 128;
- identical evaluation batches at all checkpoints and seeds.

Report CE for every checkpoint/seed. Route-utilization entropy and dead-page fraction are also recorded once; they are routing properties and should match G6a.

## Primary diagnostic criterion

The **training-exposure hypothesis is supported by G6b** only if all of the following hold:

1. the 120-step replication integrity assertion passes for all seeds;
2. for every seed, `CE(N=256, 1920 steps)` is lower than that seed's frozen G6a `CE(N=16, 120 steps)`;
3. for every seed, `CE(N=256, 1920 steps)` is lower than its own replicated `CE(N=256, 120 steps)`;
4. all frozen inference-resource assertions remain exact.

Checkpoint-to-checkpoint monotonicity is reported but is not required by the primary diagnostic; optimizer trajectories can be noisy.

If any seed fails criterion 2 or 3, the exposure hypothesis is **not supported by this diagnostic**.

## Interpretation boundary

Even if supported, G6b does not convert G6a into a pass and does not establish a fixed-training-cost capacity scaling law. It would show instead that the larger-model failure is substantially recoverable when training exposure increases with external capacity.

A supported G6b would motivate a separately predeclared G6c capacity sweep whose training-budget scaling rule is fixed before execution and reported as an explicit training resource.

An unsupported G6b would shift attention toward routing granularity, operator shape/page size, insertion depth, or other scale-dependent mismatches; any such changes require a new named experiment.
