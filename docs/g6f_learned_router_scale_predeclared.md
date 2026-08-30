# G6f: predeclared learned-factorized-router larger-LM capacity gate

Status before execution: **PREDECLARED / UNRUN**

## Purpose

G6a/G6c/G6d/G6e establish the fixed-factorized-router picture at larger subword-LM scale:

- a fixed total page-training budget fails at N=256;
- a prospective paired crossover confirms sparse page exposure/optimization is a major cause;
- with explicit capacity-scaled page training, N=64 and N=256 achieve substantially lower loss, while the N=4 versus N=16 edge remains unresolved.

G6f asks a different question:

> Does the established hidden-state-only learned factorized router preserve the larger-model capacity benefit across N=16,64,256 when the page-training exposure rule is held fixed?

G6f does not tune or reinterpret the fixed-router experiments.

## Frozen model and data

Use the exact archived G6a assets:

- tokenizer SHA-256: `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`;
- backbone SHA-256: `a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049`;
- WikiText-2 raw archive SHA-256: `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.

Frozen LM:

- train-only byte-level BPE vocabulary 1,024;
- causal Transformer `d_model=96`, 3 layers, 4 heads, FF 384, context 128;
- ParamProbe insertion after block 2.

Frozen page operator:

- `96 -> 20 -> 96`, tanh, residual scale 0.15;
- 3,956 FP32 parameters / 15,824 learned bytes;
- one 16,384-byte physical block;
- 3,840 active page matrix MACs/token;
- `q=1`, exactly 16,384 logical external parameter bytes/token.

## Learned router protocol

Use the same router family and unsupervised training objective as the successful prefix-balanced/reliability-ordered G3b/G3c router, generalized only to the frozen `d_model=96` hidden states and eight maximum address factors.

Architecture:

- MLP `96 -> 64 -> 8`, GELU between layers;
- maximum eight outputs executed for every capacity;
- matrix MACs/token: `96*64 + 64*8 = 6,656`, fixed across N;
- router parameters are resident and fixed after router training.

Router training uses **training hidden states only**. No LM labels, validation loss, future-token information, realized page losses, semantic addresses, or next-token counterfactual targets are used.

Predeclared router hyperparameters:

- router seed 17;
- hidden-bank seed 2468;
- hidden bank: 120 batches x 8 sequences x 128 tokens = 122,880 hidden states, matching the hidden-state count of the established tiny-router bank;
- router steps 600;
- router batch size 1,024;
- AdamW lr `2e-3`, weight decay `1e-4`;
- feature-scaled perturbation std `0.08`;
- consistency weight `1.0`;
- composite Rényi-2 balance weight `0.2`;
- confidence weight `0.1`;
- equal balance prefixes `2,4,6,8`.

After router training, freeze the router.

Reliability ordering is computed once from a separate deterministic training-only hidden-state bank:

- 60 batches x 8 x 128 = 61,440 hidden states, matching the established tiny-router ordering-bank hidden-state count;
- ordering hidden-bank seed 97531;
- ordering perturbation seed 86420;
- same feature-scaled perturbation std `0.08`;
- sort factors by descending hard clean-vs-noisy agreement, deterministic index tie-break.

For N=16/64/256 use the leading 4/6/8 factors of that one frozen order. Every route call still executes all eight router outputs.

## Capacity and page-training protocol

Capacities: `N=16,64,256`.

Use the same G6d exposure-floor rule:

`steps(N) = max(120, ceil(7680*N/1024))`

which gives:

- N=16: 120 steps, 122,880 total routed assignments, 7,680 mean assignments/page;
- N=64: 480 steps, 491,520 assignments, 7,680/page;
- N=256: 1,920 steps, 1,966,080 assignments, 7,680/page.

Fresh page seeds: **32,33,34**, selected before execution.

For each seed:

- page initialization seed `40000 + seed`;
- training hidden-state-bank seed `50000 + seed`;
- one 1,920-minibatch frozen hidden/target bank is created and each capacity uses the appropriate prefix;
- corresponding page-prefix rows are explicitly paired by post-construction initialization;
- page optimizer AdamW lr `4e-3`, weight decay `1e-4`.

## Evaluation and routing diagnostics

Reuse the frozen G6a validation hidden-state bank: 40 batches x 8, context 128, seed 1234.

Perturbation route stability uses that same deterministic validation hidden-state bank, feature-scaled noise std `0.08`, and a dedicated noise seed **4321**. This diagnostic does not enter the primary pass criterion.

Report:

- per-seed validation CE;
- three-seed mean/sample std;
- normalized utilization entropy and dead-page fraction at N=16/64/256;
- perturbation route stability for each active address width;
- per-factor training-only reliability and final factor order;
- router parameters and matrix MACs/token;
- page parameters/MACs, q, block bytes, and logical bytes/token;
- page-training steps and routed assignments.

## Primary criterion

G6f supports the learned scalable router at this model scale only if all of the following hold:

1. the three-seed mean CE is strictly decreasing `N=16 -> 64 -> 256`;
2. N=256 beats N=16 in every fresh seed 32/33/34;
3. the frozen inference resource contract holds exactly;
4. the capacity-scaled page-training rule holds exactly;
5. router training/order uses training hidden states only as declared.

Full per-seed adjacent monotonicity is reported as a secondary robustness diagnostic, not a primary requirement.

There is **no** primary criterion that the learned router beat the fixed router, because their routing compute differs substantially (6,656 versus 768 matrix MACs/token). Any cross-router loss comparison is descriptive rather than total-compute matched.

## Interpretation boundary

If G6f passes, it supports the claim that the larger-model large-capacity benefit is compatible with a learned factorized router whose resident metadata and active routing compute remain independent of N within the sweep.

If G6f fails, preserve the failure. Do not tune router width, loss weights, noise level, factor ordering, or page-training budgets on this frozen corpus after seeing the result. A later router follow-up, if justified, must be separately predeclared.
