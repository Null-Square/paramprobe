# ParamProbe research specification v0.7

## 1. Research question

Can learned parameter capacity increase on external storage while **worst-case external parameter traffic, resident working memory, and active inference compute remain bounded**, and does that extra inactive capacity improve learned task quality?

The central empirical hypothesis is:

> At fixed resident memory `M`, explicit external parameter traffic `Q`, and active compute `C`, task loss can improve as sparsely addressable external learned capacity `P_ext` increases.

The resource theorems define what is possible for the ParamProbe function family. They do **not** guarantee the empirical scaling hypothesis or finite-`N` superiority over conventional sparse models.

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

It also does **not** claim finite-`N` LM quality superiority over conventional flat MoE routing. Tiny Shakespeare shows the opposite at `N=64`: a task-trained flat router is lower-loss than the current factorized learned router, even with an exact page-shape match and lower active matrix compute.

The intended contribution is the **hard external parameter-probe budget as an architectural/scaling constraint**, together with scalable addressing, operator granularity, training, and physical storage.

## 7. Falsification gates

### G0 — invariants — PASSED

Reference tests cover exact factorized retrieval, file-backed/resident operator agreement, exact probe accounting, and exact logical bytes. Pinned CPU CI passes all 10 tests.

### G1/G2 — synthetic capacity, nonlinear operators, and task-only routing — PASSED WITH RETAINED NEGATIVE SUBCASES

Synthetic gates establish conditional capacity and nonlinear page expressivity under fixed probe budgets. Routing reliability/coding/local-routing failures are retained where monotonicity fails.

### G3a/G3b — LM insertion and learned-router architectural prechecks — PASSED THROUGH 64 PAGES

The successful LM insertion is `embedding -> block 1 -> ParamProbe -> block 2 -> LM head`. The standard page is `48 -> 10 -> 48`, 1,018 FP32 parameters / 4,072 learned bytes inside one 4,096-byte page, 960 active page matrix MACs/token.

A prefix-balanced, reliability-ordered causal router trained only from hidden states passes the local 1/4/16/64 architectural gate. No semantic address labels, realized next-token page targets, validation loss, or future-token information are used to train/order it.

### G3c/G3f — canonical Tiny Shakespeare capacity — FIXED ROUTER PASSES; LEARNED 256 BOUNDARY PERSISTS

Verified corpus: 1,115,394 bytes, Git blob SHA-1 `7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9`.

Under `q=1`, `B=4096`, paired fixed-hash CE for 1/4/16/64/256 pages is:

`2.46687950 / 2.46569404 / 2.46455893 / 2.46409338 / 2.46395130`.

Every seed is strictly monotone.

Paired learned-router CE is:

`2.46687950 / 2.46534705 / 2.46422635 / 2.46244999 / 2.46245038`.

Two of three seeds regress at `64 -> 256`; the mean is effectively a plateau.

### G3g — compute-matched dense control — PARAMPROBE LOWER LOSS

A resident dense adapter with 4,512 matrix MACs/token reaches `2.46482733`; learned ParamProbe uses 4,544 router+page MACs and reaches `2.46244999` at 64 pages. Comparable dense active compute does not explain the capacity gain.

### G3i/G3j — flat MoE controls — PARAMPROBE DOES NOT WIN FINITE-N QUALITY

A compute-matched 64-way flat MoE reaches `2.46123072` at 64 experts. More strictly, G3j uses the exact ParamProbe `48 -> 10 -> 48`, 4,072-byte expert and only 4,032 total matrix MACs/token—512 fewer than learned ParamProbe—and reaches `2.46182543`, still lower than learned ParamProbe's `2.46244999`.

This is a genuine constraint. Flat routing uses resident metadata linear in maximum `N` and these controls keep experts resident, so they do not satisfy the intended scalable resident-memory / hard external-parameter-probe contract, but they are currently better finite-`N` routers.

### G3h — page granularity — 16 KiB SWEET SPOT IN THIS SETUP

At fixed 1 MiB external capacity with one probe, mean CE is `2.46392645` for 4 KiB / 256 pages, `2.46253466` for 16 KiB / 64 pages, and `2.46337525` for 64 KiB / 16 pages. 16 KiB wins all three seeds.

The complete **frozen** Tiny Shakespeare controls, exact seed tables, workflow ids, and artifact digests are in `docs/g3d_j_tinyshakespeare_controls.md`.

### G4 — physical storage — BACKEND IMPLEMENTED, DEVICE STUDY PENDING

Linux `O_DIRECT + preadv` support exists. Publication-grade G4 requires named devices, repeated trials, queue-depth control, CPU measurements, end-to-end operator timing, and serialized/file-backed execution of trained LM pages.

## 8. Implementation policy

- Report external bytes, block bytes, selected pages, logical bytes read, resident routing metadata, routing compute, active operator compute, and I/O mode.
- Training may keep page tables resident, but inference claims require separately serializable selected-page execution.
- Retain negative results.
- Tiny Shakespeare is now **frozen**; do not tune its learned-router boundary post hoc.

## 9. Current publication boundary

> On canonical Tiny Shakespeare, inactive external page capacity can improve validation loss under a strict one-page external-parameter traffic budget and fixed active page compute. A paired fixed router scales monotonically through 256 pages. The present learned factorized router is strong through 64 pages but plateaus at 256, and conventional finite flat MoE routing remains lower-loss at `N=64`.

This is **not yet Q1-ready evidence**. The remaining blockers are concrete: reproduce the capacity effect on at least a second named language corpus; execute trained LM pages through the serialized/file-backed storage path under the same measured probe contract; and demonstrate the effect on a larger or more standard language-model setting.