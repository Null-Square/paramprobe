# G9: predeclared one-block operator-vs-key/value-memory challenge

Status before execution: **PREDECLARED / UNRUN**

## Purpose

G7b shows that a balanced PEER-style product-key router can satisfy the same finite q=1 / 16 KiB block contract as ParamProbe and outperform the current fixed factorized router at N=256. G8 separately confirms the factorized router's better asymptotic address-resource scaling.

G9 therefore removes routing as the comparison variable and tests the remaining payload-level question:

> Under one fetched 16 KiB learned block/token, does storing a complete nonlinear micro-operator provide a useful quality-per-probe point relative to a dense local key/value memory that uses essentially the same bytes and active compute?

G9 is not a literal DSE or SCONE reproduction. The key/value block is a deliberately strong block-adapted embedding-memory baseline designed to satisfy the exact same physical q=1 contract.

## Frozen assets and global router

Use the exact archived G6a WikiText-2 assets:

- tokenizer SHA-256 `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`;
- backbone SHA-256 `a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049`;
- official WikiText-2 raw archive SHA-256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.

Use N=256 blocks and the **same frozen balanced product-key d_key=6 router** for both payload types within each seed.

Router training is exactly the G7b label-free protocol:

- 600 router-pretraining steps;
- batch 1,024 from the fixed 81,920 train-hidden-state bank;
- AdamW lr 2e-3, weight decay 1e-4;
- perturbation noise std 0.08;
- consistency / balance / confidence weights 1.0 / 0.2 / 0.1;
- router is frozen before external-block payload training;
- inference router resources remain 672 MACs/token, 2,712 learned bytes + 56 bytes BatchNorm buffers.

Fresh seeds: **38, 39, 40**. Router seed is the ordinary G7b ProductKeyRouter seed argument equal to the experiment seed; router-pretraining RNG uses `71000 + seed` exactly as in G7b.

## Shared hard inference contract

Both payload methods use:

- N=256 external physical blocks;
- q=1 block/token;
- B=16,384 bytes;
- exactly 16,384 logical external parameter bytes/token;
- one reusable 16 KiB external-block workspace;
- the same selected block id and global router gate for a given token/seed;
- residual scale 0.15;
- frozen backbone.

Thus routing, external block count, block traffic, and backbone are paired. Only the learned contents and local computation inside the fetched block differ.

## Payload A: nonlinear operator block

Current ParamProbe page:

- `96 -> 20 -> 96` residual MLP;
- tanh inner and output nonlinearity;
- 3,956 FP32 parameters;
- 15,824 learned bytes;
- 560 padding bytes inside the 16,384-byte physical block;
- active matrix MACs/token: `96*20 + 20*96 = 3,840`.

Initialization/training are unchanged from G7b/G6:

- deterministic page initialization with `40000 + seed`;
- first-layer weights Normal(`0`, `0.15/sqrt(96)`), first bias zero, second-layer weights/bias zero;
- AdamW lr 4e-3, weight decay 1e-4.

## Payload B: dense local key/value-memory block

Each physical block contains **21 key vectors and 21 value vectors**, each 96-dimensional FP32:

- learned parameters/block: `21 * 96 * 2 = 4,032`;
- learned payload bytes: `4,032 * 4 = 16,128`;
- padding: 256 bytes;
- therefore this baseline receives **304 more learned bytes/block** than the operator.

For hidden state `x` and the selected external block:

1. score all 21 local keys by dot product `s_j = <x, k_j>`;
2. compute `a = softmax(s)` over the 21 in-block slots;
3. output `v = sum_j a_j value_j`;
4. residual is `0.15 * tanh(v)`;
5. multiply by the same frozen global product-key router gate used by Payload A.

Active matrix/dot MACs/token:

- key scoring: `21*96 = 2,016`;
- weighted value sum: `21*96 = 2,016`;
- total: **4,032 MACs/token**.

Thus the KV baseline has slightly **more** learned payload bytes and slightly **more** active matrix/dot compute than the nonlinear operator, while using the same one physical block/token.

Initialization:

- RNG seed `81000 + seed`;
- keys Normal(`0`, `1/sqrt(96)`);
- values exactly zero, giving zero initial residual like the operator's zero-initialized second layer;
- AdamW lr 4e-3, weight decay 1e-4.

No resident per-block local index is allowed; all 21 keys/values live inside the fetched external block.

## Paired training

For each fresh seed 38/39/40:

1. train one balanced product-key router from training hidden states only and freeze it;
2. materialize one page-training hidden/target bank of 1,920 minibatches x 8 sequences x 128 tokens using seed `50000 + seed`;
3. train the operator payload table on that bank;
4. train the KV payload table on the exact same bank;
5. the backbone/router remain frozen throughout payload training.

Mean routed assignments/block are approximately 7,680, matching G6c/G7/G7b.

## Evaluation

Use the frozen G6/G7 validation bank:

- official WikiText-2 validation split;
- 40 batches x 8 sequences;
- context 128;
- seed 1234.

Report per seed/method:

- validation CE;
- shared global block-utilization entropy and dead-block fraction;
- shared router gate mean;
- KV normalized local-attention entropy (mean entropy of the 21-way local softmax divided by log 21);
- payload learned bytes/block;
- payload active MACs/token;
- router bytes/MACs;
- q and logical external bytes/token.

## Predeclared classification

1. If the KV block has lower CE than the nonlinear operator in **all three seeds and in mean**, classify `operator_specific_advantage_killed`.
2. If the nonlinear operator has lower CE in **all three seeds and in mean**, classify `nonlinear_operator_supported`.
3. Otherwise classify `operator_vs_kv_unresolved`.
4. No utilization gate is needed for the global router because the exact same frozen router is shared by both payload types; nevertheless its utilization metrics are reported.
5. The KV baseline is deliberately slightly advantaged in learned bytes and active MACs. Therefore an operator win is meaningful; a KV win directly weakens the claim that complete nonlinear micro-operators are the important payload choice.
6. No result is described as DSE/SCONE superiority or inferiority; this is a resource-matched block-level mechanism comparison only.
7. No G7b or G8 conclusion is retroactively changed.

No seed, payload shape, local slot count, initialization, optimizer, training duration, router protocol, or interpretation may be changed after G9 execution begins.