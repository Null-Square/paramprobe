# ParamProbe research specification v1.2

## 1. Research question

Can learned parameter capacity increase on external storage while **worst-case external parameter traffic, resident working memory, and active inference compute remain bounded**, and does that extra inactive capacity improve learned task quality?

The central empirical hypothesis is:

> At fixed resident memory `M`, explicit external parameter traffic `Q`, and active inference compute `C`, task loss can improve as sparsely addressable external learned capacity `P_ext` increases.

The resource theorems define what is possible for the ParamProbe function family. They do **not** imply that every individual training run must improve at every discrete capacity step, nor do they imply finite-`N` superiority over conventional sparse models.

A separate empirical question is how **training sample/optimization cost** must scale with external capacity. ParamProbe's bounded-inference claim does not imply bounded training cost.

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
- **C7:** training exposure/optimization is a separate scaling resource. Keeping inference `M`, `C`, and `Q` bounded does not guarantee that a fixed number of training token assignments is sufficient as `N` grows.

## 6. Non-claims

ParamProbe does not claim to invent model offloading, sparse MoE, Product-Key routing, learned sparse memory, SSD-resident parameters, direct-I/O expert storage, error-correcting output codes, counterfactual router training, or collision/Rényi load balancing.

It does **not** claim a deterministic monotone scaling law in every individual seed or at every adjacent capacity step. Small reversals occur in the tiny WikiText-2 experiments, and the larger subword experiment contains a near-flat/unresolved `N=4` versus `N=16` region.

It does **not** claim that bounded inference resources imply bounded training compute or sample complexity. G6a shows that a fixed page-training budget can fail as external capacity grows; G6c shows that substantially increasing page-training exposure can reverse that failure without changing inference resources.

It does **not** claim capacity gains at fixed total training compute in the larger-model setting. G6d explicitly allows page-training work to grow with capacity.

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

Across combined seeds `7..19`, fixed-routing mean CE is:

`2.48249533 / 2.48209350 / 2.48130979 / 2.48081780 / 2.48003056`.

- 256 pages beats one page in 13/13 seeds;
- 11/13 seeds are strictly monotone over all four adjacent capacity steps;
- paired mean `1 -> 256` change is `-0.00246477` CE, approximate 95% Student-t interval `[-0.00262678, -0.00230275]`.

Learned-routing mean CE is:

`2.48249533 / 2.48128070 / 2.47973229 / 2.47903057 / 2.47877004`.

- 256 pages beats one page in 13/13 seeds;
- 12/13 seeds are strictly monotone over all four adjacent steps;
- paired mean `64 -> 256` change is `-0.00026053` CE, approximate 95% interval `[-0.00036150, -0.00015955]`;
- paired mean `1 -> 256` change is `-0.00372529` CE, approximate 95% interval `[-0.00389656, -0.00355401]`.

Full details are in `docs/g5b_wikitext2_robustness.md`.

### G6a — larger subword-LM fixed-total-training scale smoke — FAILED

G6a changes model/tokenization scale rather than tuning the tiny setup:

- train-only byte-level BPE vocabulary 1,024;
- 446,304-parameter causal Transformer, `d_model=96`, 3 layers, context 128;
- insertion after block 2;
- fixed balanced 8-factor hash;
- page operator `96 -> 20 -> 96`, 3,956 FP32 parameters / 15,824 learned bytes;
- one 16,384-byte physical block/token (`q=1`);
- active page matrix compute 3,840 MACs/token;
- fixed router matrix compute 768 MACs/token;
- paired capacity points `N=1,16,256`, seeds 7/8/9;
- fixed page-training schedule: 120 minibatches of `8 x 128` tokens for every capacity.

Mean CE for `N=1/16/256` is:

`5.70263753 / 5.70234172 / 5.70345494`.

All three seeds improve at `1 -> 16`, then all three regress at `16 -> 256`; 256 pages is worse than one page in all three seeds. Resource/leakage assertions pass; N=256 utilization entropy is `0.90570751` and dead-page fraction is zero.

The fixed training schedule supplies 122,880 routed token assignments total: roughly 122,880/page at N=1, 7,680/page at N=16, and only 480/page at N=256. G6a remains frozen as a scientific failure. See `docs/g6a_subword_scale_result.md`.

### G6b — first exposure diagnostic — INVALID / NOT SCIENTIFIC EVIDENCE

G6b extended the frozen G6a N=256 page-training trajectory to 1,920 steps and produced a strongly improving apparent curve. However, its predeclared 120-step cross-run replication tolerance of `2e-7` failed by small amounts, and the original workflow also exposed a shell-pipeline issue in which `tee` could mask the Python exit code.

Per the predeclared integrity rule, G6b is classified as protocol/infrastructure-invalid. Its favorable trajectory is **not counted** as scientific evidence and its tolerance was not relaxed after observing the result. See `docs/g6b_training_exposure_result.md`.

### G6c — independent paired exposure crossover — PASSED

G6c removes the cross-run equality dependency and uses fresh page seeds 10/11/12. Within each run/seed, N=16 and N=256 receive paired hidden-state/target minibatch prefixes from the exact frozen G6a assets.

At 120 page-training steps, N=16 beats N=256 in all three fresh seeds. N=256 then trains continuously to 1,920 steps, at which point both capacities have the same **mean routed assignments/page** (7,680), although N=256 has used 16x more total page-training assignments/optimization work.

Per-seed CE:

| seed | N=16 @ 120 | N=256 @ 120 | N=256 @ 1920 |
|---:|---:|---:|---:|
| 10 | 5.70218958 | 5.70326931 | 5.69937409 |
| 11 | 5.70235314 | 5.70332602 | 5.69955056 |
| 12 | 5.70200230 | 5.70325141 | 5.69923952 |

N=256 mean CE falls monotonically across exposure checkpoints:

`5.70328225 / 5.70176050 / 5.70061757 / 5.69984304 / 5.69938806`

for 120/240/480/960/1920 steps.

The paired N=16 mean is `5.70218167`; final N=256 improves on it by `-0.00279361` CE. Every predeclared crossover condition passes in every fresh seed.

G6c therefore confirms that sparse page training exposure/optimization is a **major cause** of the G6a 256-page failure. It does not make G6a pass and does not establish fixed-total-training-compute capacity scaling. See `docs/g6c_paired_exposure_crossover_result.md`.

### G6d — full capacity sweep with explicit training-budget scaling — PRIMARY GATE FAILED; LARGE-N BENEFIT RETAINED

G6d predeclares the page-training rule

`steps(N) = max(120, ceil(7680*N/1024))`,

which gives 120/120/120/480/1920 steps for `N=1/4/16/64/256`. Inference resources remain fixed; total page-training work grows beyond 16 pages.

Fresh seeds 13/14/15 give mean CE:

`5.70268902 / 5.70239363 / 5.70243641 / 5.70103632 / 5.69948625`.

The predeclared gate required a strictly decreasing mean across all five capacities. It **fails** because of a small `4 -> 16` mean reversal of about `+0.00004278` CE.

The larger-capacity effects are much larger: `16 -> 64` improves by about `-0.00140009` and `64 -> 256` by about `-0.00155007`. N=256 beats both N=1 and N=16 in all three fresh seeds. Resource and training-budget-rule assertions pass.

G6d remains frozen as a failed strict monotonicity gate. See `docs/g6d_capacity_scaled_training_result.md`.

### G6e — paired N=4 versus N=16 robustness — UNRESOLVED

G6e isolates the only G6d mean reversal using 16 fresh paired page seeds 16..31 under the exact same 120-step protocol for both capacities.

Mean CE is `5.70229593` for N=4 and `5.70234439` for N=16. The paired effect `CE16 - CE4` is:

`+0.00004846 +/- 0.00014011` sample standard deviation,

with two-sided 95% Student-t interval:

`[-0.00002621, +0.00012312]`.

N=16 is lower-loss in 5/16 seeds. The interval overlaps zero, so the predeclared classification is **unresolved**. This does not reclassify the frozen G6d failure. See `docs/g6e_low_capacity_edge_result.md`.

The combined larger-model interpretation is therefore:

> Under a fixed total page-training budget, 256-page scaling fails. A prospective paired crossover confirms that inadequate sparse page exposure is a major cause. When page-training work is allowed to scale explicitly with capacity, 64 and 256 pages provide substantially lower loss while the inference traffic/router/page-compute envelope remains fixed; the low-capacity N=4 versus N=16 edge remains statistically unresolved rather than reliably monotone.

## 8. Implementation and evidence policy

- Report external bytes, block bytes, selected pages, logical bytes read, resident routing metadata, routing compute, active operator compute, and I/O mode.
- Report training token assignments/optimization budget separately from inference-resource claims whenever capacity changes.
- Distinguish fixed-total-training-compute experiments from capacity-scaled-training experiments explicitly.
- Training may keep page tables resident, but inference claims require separately serializable selected-page execution.
- Retain negative results and infrastructure-only failures separately from scientific results.
- Tiny Shakespeare, original three-seed WikiText-2 G5, G6a, and G6d pass/fail outcomes are frozen; do not tune or retroactively redefine them.
- G6b remains invalid and must not be counted as evidence.
- Additional seed/optimization studies cannot retroactively change earlier pass/fail criteria.
- Any larger-model or learned-router follow-up must be separately predeclared with model/tokenizer/page/router/training-budget changes explicit.
- Virtualized/cloud storage timing is diagnostic unless the physical storage stack is sufficiently characterized.

## 9. Current publication boundary

> Across the tiny byte-level language-model studies, inactive external page capacity improves validation loss under a hard one-page inference traffic budget on Tiny Shakespeare and WikiText-2, with strong multi-seed robustness on WikiText-2. Trained pages also execute through real `pread` and `O_DIRECT` fixed-block storage with numerically equivalent outputs and exactly one page of explicit parameter traffic per token. In the larger 446k-parameter, subword-tokenized WikiText-2 model, a fixed total page-training budget fails at N=256, but an independent paired experiment confirms that increasing sparse page-training exposure reverses this failure. Under an explicit capacity-scaled training rule, N=64 and N=256 yield substantially lower loss and N=256 beats N=1 and N=16 in every fresh seed while inference traffic and active page compute remain fixed. A small N=4-versus-N=16 edge is statistically unresolved, so the evidence supports a large-capacity trend conditional on training-resource scaling rather than a universal monotone law or a fixed-training-compute scaling law.

This is **not yet Q1-ready evidence**. The major remaining scientific questions are now: characterize the training-resource/quality scaling law more efficiently; test whether a learned scalable router preserves the larger-model capacity benefit; and evaluate the resulting quality/latency/capacity tradeoff with real file-backed pages. Publication-grade bare-metal storage validation remains a separate systems blocker.
