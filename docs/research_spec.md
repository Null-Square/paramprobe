# ParamProbe research specification v1.0

## 1. Research question

Can learned parameter capacity increase on external storage while **worst-case external parameter traffic, resident working memory, and active inference compute remain bounded**, and does that extra inactive capacity improve learned task quality?

The central empirical hypothesis is:

> At fixed resident memory `M`, explicit external parameter traffic `Q`, and active compute `C`, task loss can improve as sparsely addressable external learned capacity `P_ext` increases.

The resource theorems define what is possible for the ParamProbe function family. They do **not** imply that every individual training run must improve at every discrete capacity step, nor do they imply finite-`N` superiority over conventional sparse models.

## 2. Parameter-probe model

A `q`-probe layer over `B`-byte external parameter blocks satisfies `sup_x Pi_B(f,x) <= q`, so logical explicit parameter traffic is bounded by `Q <= qB` per invocation, independent of total external capacity `NB`. Physical device traffic is measured separately.

## 3. Page-sized conditional operator

A resident router selects at most `q` complete external micro-operators. Each selected operator fits inside one block and can be streamed through a reusable `B`-byte workspace. Thus external-operator workspace does not scale with `N`.

`B` is an architectural hyperparameter: larger pages provide more local operator capacity but cost more bytes/probe and can reduce address count at fixed total storage.

## 4. Scalable addressing

Flat `N`-way routing is excluded from the asymptotic claim because it hides `Theta(N)` resident metadata. For factorized `N=m^r` addressing with constant radix/key widths, resident routing metadata grows as `Theta(log N)` while external learned capacity grows as `Theta(NB)`.

For finite sweeps claiming fixed routing resources, maximum address width is allocated once and smaller capacities use prefixes/masks of that same output.

## 5. Core claims and constraints

- **C1:** logical external traffic is at most `qB`, independent of `N`.
- **C2:** sequential page execution needs approximately `B + O(d+q)` external-operator workspace.
- **C3:** factorized router metadata can grow sublinearly in total external capacity.
- **C4:** hard dense function families provide a contrast showing constant external probes with tiny auxiliary state are not universal.
- **C5:** routing reliability is a separate scaling resource; multi-factor address success can decay as address width grows.
- **C6:** composite-address collision/Rényi balancing can be estimated using only factor probabilities, but balancing does not determine page utility.

## 6. Non-claims

ParamProbe does not claim to invent model offloading, sparse MoE, Product-Key routing, learned sparse memory, SSD-resident parameters, direct-I/O expert storage, error-correcting output codes, counterfactual router training, or collision/Rényi load balancing.

It does **not** claim a deterministic monotone scaling law in every individual training seed. Small adjacent-capacity reversals occur on WikiText-2 even though the endpoint capacity effect is sign-consistent across all evaluated seeds.

It also does **not** claim finite-`N` LM quality superiority over conventional flat MoE routing. Tiny Shakespeare shows the opposite at `N=64`: a task-trained flat router is lower-loss than the current factorized learned router, even with an exact page-shape match and lower active matrix compute.

The intended contribution is the **hard external parameter-probe budget as an architectural/scaling constraint**, together with scalable addressing, operator granularity, compatible training, and physical storage execution.

## 7. Falsification gates

### G0 — invariants — PASSED

Reference tests cover exact factorized retrieval, file-backed/resident operator agreement, exact probe accounting, and exact logical bytes. Pinned CPU CI passes all 10 tests.

### G1/G2 — synthetic capacity, nonlinear operators, and task-only routing — PASSED WITH RETAINED NEGATIVE SUBCASES

Synthetic gates establish conditional capacity and nonlinear page expressivity under fixed probe budgets. Routing reliability/coding/local-routing failures are retained where monotonicity fails.

### G3a/G3b — LM insertion and learned-router architectural prechecks — PASSED THROUGH 64 PAGES

The successful tiny-LM insertion is `embedding -> block 1 -> ParamProbe -> block 2 -> LM head`. The standard page is `48 -> 10 -> 48`, 1,018 FP32 parameters / 4,072 learned bytes inside one 4,096-byte page, 960 active page matrix MACs/token.

A prefix-balanced, reliability-ordered causal router trained only from hidden states passes the local `1/4/16/64` architectural gate. No semantic address labels, realized next-token page targets, validation loss, or future-token information are used to train/order it.

### G3c/G3f — canonical Tiny Shakespeare capacity — FIXED ROUTER PASSES; LEARNED 256 PLATEAU RETAINED

Verified corpus: 1,115,394 bytes, Git blob SHA-1 `7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9`.

Under `q=1`, `B=4096`, paired fixed-hash CE for `1/4/16/64/256` pages is:

`2.46687950 / 2.46569404 / 2.46455893 / 2.46409338 / 2.46395130`.

Every seed is strictly monotone.

Paired learned-router CE is:

`2.46687950 / 2.46534705 / 2.46422635 / 2.46244999 / 2.46245038`.

Two of three seeds regress slightly at `64 -> 256`; the mean is effectively a plateau. This result is frozen rather than tuned away.

### G3g — compute-matched dense control — PARAMPROBE LOWER LOSS

A resident dense adapter with 4,512 matrix MACs/token reaches `2.46482733`; learned ParamProbe uses 4,544 router+page MACs and reaches `2.46244999` at 64 pages. Comparable dense active compute does not explain the capacity gain.

### G3i/G3j — flat MoE controls — PARAMPROBE DOES NOT WIN FINITE-N QUALITY

A compute-matched 64-way flat MoE reaches `2.46123072` at 64 experts. More strictly, G3j uses the exact ParamProbe `48 -> 10 -> 48`, 4,072-byte expert and only 4,032 total matrix MACs/token—512 fewer than learned ParamProbe—and reaches `2.46182543`, still lower than learned ParamProbe's `2.46244999`.

This is a genuine constraint. Flat routing uses resident metadata linear in maximum `N` and these controls keep experts resident, so they do not satisfy the intended scalable resident-memory / hard external-parameter-probe contract, but they are currently better finite-`N` routers.

### G3h — page granularity — 16 KiB SWEET SPOT IN THIS SETUP

At fixed 1 MiB external capacity with one probe, mean CE is `2.46392645` for 4 KiB / 256 pages, `2.46253466` for 16 KiB / 64 pages, and `2.46337525` for 64 KiB / 16 pages. 16 KiB wins all three seeds.

The complete frozen Tiny Shakespeare controls, seed tables, workflow ids, and artifact digests are in `docs/g3d_j_tinyshakespeare_controls.md`.

### G4a — direct-I/O backend precheck — PASSED AS IMPLEMENTATION CHECK

Linux `O_DIRECT + preadv` support bypasses the ordinary OS page cache for explicit fixed-block parameter reads. The random-byte block-size sweep establishes that the direct-I/O code path is functional and that page granularity materially changes latency/throughput, but those virtualized measurements are not hardware claims.

### G4b — trained LM file-backed execution — PASSED FUNCTIONAL STORAGE GATE

A trained 64-page Tiny Shakespeare learned-router model is serialized into exact 4 KiB blocks and executed through both explicit `pread` and aligned `O_DIRECT + preadv` storage.

On a deterministic 256-token validation batch, resident, serialized-block, `pread`, and `O_DIRECT` CE are all `2.52785110`; maximum residual mismatch is `2.98e-8`. Both file-backed paths execute exactly 256 probes / 1,048,576 bytes = 4,096 explicit parameter bytes/token.

A systems-only mirrored-store sweep expands the backing file from 0.25 MiB to 256 MiB while preserving the learned function. Process RSS remains flat. The run used a virtualized Azure `MSFT NVMe Accelerator v1.0`, so its latency values remain diagnostic rather than publication hardware claims. See `docs/g4b_filebacked_lm.md`.

### G4c — publication physical storage — PENDING

Publication-grade physical validation still requires named bare-metal devices, repeated independent trials, queue-depth/concurrency control, cold/warm methodology, CPU measurements, pure-I/O versus page-operator versus full-token latency, and stores comfortably larger than host cache.

### G5 — WikiText-2 raw frozen replication — MEAN TREND REPLICATES; ORIGINAL STRICT GATES FAIL

G5 changes only the corpus to the official WikiText-2 raw train/validation split. The original archive identity is hard-checked at 4,721,645 bytes / SHA-256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`. The frozen architecture, optimizer, routing protocol, paired minibatches, page shape, `q=1`, and 4 KiB/token traffic remain unchanged.

Fixed-hash three-seed mean CE for `1/4/16/64/256` pages is:

`2.48237525 / 2.48208454 / 2.48137037 / 2.48090410 / 2.48012896`.

Seed 8 has a small local `1 -> 4` reversal, so the predeclared every-step/every-seed fixed gate does not pass.

Learned-router three-seed mean CE is:

`2.48237525 / 2.48138192 / 2.47995500 / 2.47904339 / 2.47891196`.

Seed 7 has a tiny `64 -> 256` reversal, so the original strict learned gate also does not pass. These outcomes are frozen. See `docs/g5_wikitext2_capacity.md`.

### G5b — WikiText-2 predeclared robustness seeds — ROBUST CAPACITY TREND

G5b was declared only after G5 and adds ten independent page-training seeds (`10..19`) under the unchanged protocol. It is an estimation study, not a retroactive redefinition of G5.

Across the combined 13 seeds `7..19`, fixed-routing mean CE is:

`2.48249533 / 2.48209350 / 2.48130979 / 2.48081780 / 2.48003056`.

- 256 pages beats one page in 13/13 seeds;
- 11/13 seeds are strictly monotone over all four adjacent capacity steps;
- stepwise improvement counts are `11/13, 13/13, 13/13, 13/13`;
- paired mean `1 -> 256` change is `-0.00246477` CE, approximate 95% Student-t interval `[-0.00262678, -0.00230275]`.

Learned-routing mean CE is:

`2.48249533 / 2.48128070 / 2.47973229 / 2.47903057 / 2.47877004`.

- 256 pages beats one page in 13/13 seeds;
- 12/13 seeds are strictly monotone over all four adjacent steps;
- stepwise improvement counts are `13/13, 13/13, 13/13, 12/13`;
- paired mean `64 -> 256` change is `-0.00026053` CE, approximate 95% interval `[-0.00036150, -0.00015955]`;
- paired mean `1 -> 256` change is `-0.00372529` CE, approximate 95% interval `[-0.00389656, -0.00355401]`.

The isolated G5 learned seed-7 reversal is therefore better interpreted as a small stochastic local reversal than as evidence for a universal 256-page wall on WikiText-2. Tiny Shakespeare's 256-page learned plateau remains a separate frozen constraining result. Full details are in `docs/g5b_wikitext2_robustness.md`.

## 8. Implementation and evidence policy

- Report external bytes, block bytes, selected pages, logical bytes read, resident routing metadata, routing compute, active operator compute, and I/O mode.
- Training may keep page tables resident, but inference claims require separately serializable selected-page execution.
- Retain negative results and infrastructure-only failures separately from scientific results.
- Tiny Shakespeare and the original three-seed WikiText-2 G5 gate are frozen; do not tune either post hoc.
- Additional seed studies cannot retroactively change earlier pass/fail criteria.
- Any larger-model gate must predeclare model scale, tokenization, insertion point, page size, capacity sweep, router resources, optimizer, seeds, and pass criteria before observing its capacity results.
- Virtualized/cloud storage timing is diagnostic unless the physical storage stack is sufficiently characterized.

## 9. Current publication boundary

> Across canonical Tiny Shakespeare and WikiText-2 raw, inactive external page capacity improves mean validation loss under a strict one-page external-parameter traffic budget and fixed active page compute. On WikiText-2, the endpoint benefit is sign-consistent across all 13 evaluated page-training seeds for both fixed and learned factorized routing; learned routing is strictly monotone through 256 pages in 12/13 seeds. This is evidence for a robust capacity trend, not a deterministic per-seed scaling law. Conventional finite flat MoE routing remains lower-loss at `N=64` in the tiny benchmark. Separately, trained LM pages execute through real `pread` and `O_DIRECT` fixed-block storage with numerically equivalent outputs and exactly one page of explicit parameter traffic per token.

This is **not yet Q1-ready evidence**. The second-corpus robustness blocker is now substantially reduced. The largest remaining empirical blockers are: demonstrate the effect in a materially larger, subword-tokenized LM setting; run the trained-page storage path on characterized bare-metal hardware; and then evaluate whether the quality/latency/capacity tradeoff remains compelling at realistic scale.
