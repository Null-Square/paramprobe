# G9b predeclaration: final balanced local-KV challenge

Status before implementation/execution: **PREDECLARED / UNRUN**

## Purpose

G9's one-block nonlinear operator beat a resource-matched 21-slot local KV block in every seed. G9a then showed that the tested KV training protocol genuinely collapsed: on ~1.97M training queries, exactly one of 21 local slots was used inside each observed global block, leaving 95.238% of the 5,376 `(global block, local slot)` pairs dead.

G9a pre-authorized exactly **one** training-only anti-collapse follow-up. G9b is that final follow-up. There will be no G9c and no further KV tuning after this experiment, regardless of outcome.

The purpose of G9b is deliberately adversarial:

> Give the local KV baseline a separate label-free balanced/stable local-key partitioning phase before task training, without changing any inference resource, then ask whether the nonlinear operator still wins.

## Frozen assets and global routing

Use the exact archived G6a assets and checksum-pinned WikiText-2 raw data:

- tokenizer SHA-256 `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`;
- backbone SHA-256 `a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049`;
- G6a artifact SHA-256 `0e901efe5d360b905fbb15c5cdb5d90ffea3ad60d7f05e2712fe43fc37d2ad21`;
- WikiText-2 archive SHA-256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.

Fresh experiment seeds: **44, 45, 46**.

For each seed, train exactly one balanced product-key `d_key=6` global router using the frozen G7b label-free protocol, then freeze it. The same frozen global router is shared by the nonlinear operator and balanced-KV payloads for that seed.

Global-router inference resources remain:

- 2,712 learned parameter bytes + 56 bytes BatchNorm buffers;
- 672 routing matrix/dot MACs/token;
- no change to q or external payload traffic.

## Frozen hard inference contract

Both payload methods use:

- `N = 256` external physical blocks;
- `q = 1` block/token;
- `B = 16,384` bytes;
- exactly 16,384 logical external parameter bytes/token;
- one reusable 16 KiB external-block workspace;
- identical selected global block id and identical global router gate for a token/seed;
- frozen backbone;
- residual scale 0.15.

### Nonlinear operator payload

Unchanged G9 operator:

- `96 -> 20 -> 96` tanh residual MLP;
- 3,956 FP32 parameters/block;
- 15,824 learned bytes/block;
- 560 padding bytes;
- 3,840 active matrix MACs/token;
- initialization seed `40000 + seed`;
- AdamW task training lr `4e-3`, weight decay `1e-4`.

### Balanced local-KV payload

Inference mechanism is **identical to G9**:

- 21 local 96-D keys + 21 local 96-D values per block;
- 4,032 FP32 parameters/block;
- 16,128 learned bytes/block;
- 256 padding bytes;
- 4,032 active key-score/weighted-value MACs/token;
- raw local dot products `<x,k_j>`;
- 21-way local softmax;
- weighted value sum;
- residual `0.15 * tanh(v)` multiplied by the same frozen global router gate.

No normalization layer, resident local index, additional projection, extra inference lookup, or extra inference parameter is allowed. G9b changes **training only**.

## Stage A: label-free local-key anti-collapse pretraining

After the global router is trained and frozen, initialize the KV keys/values exactly as in G9 using RNG seed `81000 + seed`:

- keys Normal(`0`, `1/sqrt(96)`);
- values exactly zero.

Values remain zero and receive no gradients during Stage A.

Build a fixed **training-hidden-state-only local-key bank**:

- official WikiText-2 training split only;
- frozen backbone;
- 320 batches x 8 sequences x 128 tokens = **327,680 hidden states**;
- minibatch sampling seed **97531**;
- no targets, next-token labels, validation data, or validation loss are used for local-key pretraining.

Route this bank once with the frozen global product-key router and group hidden states by selected global block.

For each of **600 local-key pretraining steps**:

1. For every global block having at least one hidden state in the fixed bank, sample **16 hidden states with replacement** from that block. Sampling RNG is `83000 + seed`.
2. Let `feature_scale` be the per-feature population standard deviation over the complete 327,680-state local-key bank, clamped below at `1e-4`.
3. Form two perturbed views of each sampled hidden state with independent Gaussian noise `0.08 * feature_scale`.
4. Within the already-selected global block, compute the ordinary G9 21-way local softmax over raw key dot products for each perturbed view: `p_a`, `p_b`.
5. Let `p_bar = 0.5 * (p_a + p_b)`.
6. Compute:
   - consistency = `21 * mean((p_a - p_b)^2)`;
   - for each represented global block, local marginal = mean of `p_bar` across its 16 samples;
   - balance = mean over represented blocks of `log(21) - H(local marginal)`;
   - confidence = mean query entropy `H(p_bar) / log(21)`.
7. Optimize **keys only** with

   `loss = 1.0 * consistency + 0.2 * balance + 0.1 * confidence`.

The weights `1.0 / 0.2 / 0.1` and noise std `0.08` are inherited exactly from G7b rather than tuned on G9/G9a outcomes.

Key-pretraining optimizer:

- AdamW;
- lr `2e-3`;
- weight decay `1e-4`.

After 600 steps, **freeze all local keys permanently**. No task-loss gradient may update them.

This intentionally gives the KV baseline extra training-only compute and a dedicated label-free partitioning phase. That is allowed because the research question concerns inference resources; the stronger baseline is favored rather than handicapped.

## Stage B: paired task training

For each fresh seed 44/45/46, materialize the usual paired G9 task-training bank:

- 1,920 minibatches;
- batch 8;
- context 128;
- RNG seed `50000 + seed`;
- same bank for both payload methods.

Train:

1. the nonlinear operator exactly as in G9;
2. the balanced KV **values only**, with the Stage-A local keys frozen.

KV value optimizer:

- AdamW;
- lr `4e-3`;
- weight decay `1e-4`.

The global router and backbone remain frozen throughout both payload task-training runs.

## Evaluation and anti-collapse gate

Use the frozen G6/G7/G9 validation bank:

- official WikiText-2 validation split;
- 40 batches x 8 sequences;
- context 128;
- seed 1234.

Report CE and shared global-router diagnostics for both payloads.

For the balanced KV, also run the exact G9a hard local-slot usage diagnostic on the **full 1,920-batch task-training bank**, treating each `(global block, local slot)` pair as one of 5,376 slots.

G9b is eligible for CE interpretation only if, in **every** fresh seed:

- training normalized global hard local-slot utilization entropy >= **0.85**; and
- training dead local-slot fraction <= **0.05**.

Also report validation hard usage, per-block active fractions, max local-slot traffic shares, and mean per-query softmax entropy, but the training bank determines this health gate.

## Predeclared classification

Apply in this order:

1. If the anti-collapse utilization gate fails in any seed, classify
   `g9b_classification=anti_collapse_failed`.
   The original G9 positive result remains frozen, but no claim is made that the nonlinear operator has defeated a healthy local-KV competitor. **No further KV tuning is allowed.**
2. If utilization passes in every seed and balanced KV has lower CE than the nonlinear operator in **all three seeds and in mean**, classify
   `g9b_classification=operator_specific_advantage_killed`.
3. If utilization passes in every seed and the nonlinear operator has lower CE than balanced KV in **all three seeds and in mean**, classify
   `g9b_classification=nonlinear_operator_supported_after_balanced_kv`.
4. If utilization passes but wins are mixed, classify
   `g9b_classification=operator_vs_balanced_kv_unresolved`.

No seed, data bank, key-pretraining bank, sample count, local slot count, loss coefficient, optimizer, noise scale, training duration, inference formula, or interpretation criterion may be changed after G9b implementation/execution begins.

After G9b, the payload-comparison branch is permanently closed. The next research stage is the large external-capacity frontier, not additional KV/operator tuning.