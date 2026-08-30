# ParamProbe research specification v0.7

## 1. Research question

Can learned parameter capacity increase on external storage while **worst-case external parameter traffic, resident working memory, and active inference compute remain bounded**, and does that extra inactive capacity improve learned task quality?

The central empirical hypothesis is:

> At fixed resident memory `M`, explicit external parameter traffic `Q`, and active compute `C`, task loss can improve as sparsely addressable external learned capacity `P_ext` increases.

The resource theorems define what is possible for the ParamProbe function family. They do **not** guarantee the empirical scaling hypothesis or finite-`N` superiority over conventional sparse models.

## 2. Parameter-probe model

Partition external learned state into `N` equal blocks `P_0, ..., P_{N-1}`, each exactly `B` bytes. For an input `x`, let `Pi_B(f, x)` be the number of external parameter-block reads used while evaluating the external-memory component of `f`.

A `q`-probe layer must satisfy `sup_x Pi_B(f, x) <= q`. Therefore logical explicit parameter traffic is bounded by `Q <= qB` per invocation, independent of total external capacity `NB`.

Physical device traffic is measured separately. Logical probe counting is not a substitute for uncached/direct-I/O measurements.

## 3. Page-sized conditional operator

Given resident hidden state `h in R^d`, a resident router returns at most `q` external block ids. Each selected block encodes a complete micro-operator `E_p`. For strict probe-bound experiments, each `E_p` fits inside one block and selected pages are streamed through a reusable `B`-byte workspace, so external-operator workspace does not scale with `N`.

`B` is an architectural hyperparameter: larger pages provide greater local operator capacity and may amortize storage latency, but increase bytes per probe and can reduce the number of addressable operators at fixed total storage.

## 4. Scalable addressing

A flat `N`-way router is excluded from the asymptotic claim because it can hide `Theta(N)` resident metadata.

For the reference factorized construction, let `N=m^r`. With subkey width `s`, resident key metadata is `Theta(rms)=Theta(ms log_m N)`. The complete Cartesian address space is never materialized.

For empirical sweeps claiming strictly fixed resident router size and active routing compute, the maximum address width is allocated once and smaller capacities use prefixes/masks of that same output.

## 5. Core theoretical claims

### C1 — hard logical probe bound

If at most `q` complete block-sized operators are selected and evaluated sequentially, logical external parameter traffic is at most `qB`, independent of `N`.

### C2 — bounded external-operator workspace

Sequential page evaluation requires additional working storage of approximately `B + O(d+q)`, independent of total external page count.

### C3 — sublinear routing metadata

For `N=m^r`, factor-key metadata is `Theta(rms)`. With constant `m,s`, routing metadata grows as `Theta(log N)` while external learned capacity grows as `Theta(NB)`.

### C4 — dense-function contrast

Known cell-probe lower bounds for hard dense function families show that arbitrary dense transformations do not generally admit constant external probes with tiny auxiliary state. This is a contrast, not a claim that every conventional neural layer literally reads every stored weight on every input.

### C5 — routing reliability is a separate scaling resource

If an uncoded binary factor is correct independently with probability `p`, an `r`-bit address succeeds with probability `p^r`. Since `r=log_2 N`, address reliability can fall as the address space grows even while marginal factor accuracy is high.

### C6 — composite-address collision can be regularized in `O(log N)` routing space

For factorized Bernoulli page probabilities, complete-address collision can be estimated from products of per-factor match probabilities without materializing all `N` addresses. This controls collapse; it does not determine routing utility.

## 6. Non-claims

ParamProbe does not claim to invent model offloading, sparse MoE, Product-Key routing, learned sparse memory, SSD-resident parameters, direct-I/O expert storage, error-correcting output codes, counterfactual router training, or collision/Rényi load balancing.

It also does **not** claim finite-`N` LM quality superiority over conventional flat MoE routing. Tiny Shakespeare shows the opposite at `N=64`: a task-trained flat router is lower-loss than the current factorized learned router, even with an exact page-shape match and lower active matrix compute.

The intended contribution is the **hard external parameter-probe budget as an architectural/scaling constraint**, together with scalable addressing, operator granularity, training, and physical storage.

## 7. Falsification gates

### G0 — invariants — PASSED

Reference tests cover factorized top-k retrieval, file-backed/resident operator agreement, exact probe accounting, and exact `qB` logical bytes read. Pinned CPU CI passes all 10 repository tests.

### G1 — associative capacity — PASSED WITH ROUTING LIMITATIONS

A stateless one-probe associative memory follows the expected collision law under fixed bytes/query. Learned routing exposes an address-reliability wall; coding helps only conditionally.

### G2 — synthetic nonlinear operators/task-only routing — PASSED WITH NEGATIVE SUBCASES

The synthetic sequence establishes page-sized nonlinear expressivity, joint trainability controls, composite-address balancing, and a fixed-budget task-only counterfactual routing gate. Negative local-routing and coding results are retained.

### G3a — LM insertion precheck — PASSED AS DIAGNOSTIC

The successful insertion is `embedding -> Transformer block 1 -> ParamProbe -> Transformer block 2 -> LM head`. The page MLP is `48 -> 10 -> 48`, containing 1,018 FP32 parameters / 4,072 learned bytes in one 4,096-byte page and 960 active page matrix MACs/token.

### G3b — learned causal routing architectural precheck — PASSED THROUGH 64 PAGES

A prefix-balanced, reliability-ordered causal router trained only from hidden states passes the local 1/4/16/64 gate across three page-training seeds. No semantic address labels, realized next-token page targets, validation loss, or future-token information are used to train/order it.

### G3c/G3f — canonical Tiny Shakespeare capacity — FIXED ROUTER PASSES; LEARNED 256 BOUNDARY PERSISTS

The verified Karpathy Tiny Shakespeare corpus contains `1,115,394` bytes with Git blob SHA-1 `7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9`.

Under `q=1`, `B=4096`, the stricter paired G3f fixed-hash means are:

- 1 page: `2.46687950`;
- 4 pages: `2.46569404`;
- 16 pages: `2.46455893`;
- 64 pages: `2.46409338`;
- 256 pages: `2.46395130`.

Every fixed-hash seed is strictly monotone through 256 pages.

Paired learned-router means are:

- 1 page: `2.46687950`;
- 4 pages: `2.46534705`;
- 16 pages: `2.46422635`;
- 64 pages: `2.46244999`;
- 256 pages: `2.46245038`.

Two of three seeds regress at `64 -> 256`; the mean is effectively a plateau. The eight-factor reliability boundary is retained.

### G3g — compute-matched dense control — PARAMPROBE LOWER LOSS

A resident `48 -> 47 -> 48` adapter uses 4,512 matrix MACs/token versus 4,544 for learned ParamProbe router+page, but reaches `2.46482733` mean CE. Paired learned ParamProbe at 64 pages reaches `2.46244999`.

This supports the claim that the 64-page gain is not explained merely by comparable extra active dense compute.

### G3i/G3j — flat MoE controls — PARAMPROBE DOES NOT WIN FINITE-N QUALITY

G3i uses a 64-way flat router plus `48 -> 15 -> 48` expert for 4,512 total matrix MACs/token and reaches `2.46123072` at 64 experts.

G3j uses the exact ParamProbe `48 -> 10 -> 48`, 4,072-byte expert and only 4,032 total matrix MACs/token—512 fewer than learned ParamProbe—and reaches `2.46182543` at 64 experts, still lower than learned ParamProbe's `2.46244999`.

This is a genuine constraining result. Current factorized learned routing is not finite-`N` quality-superior to conventional flat routing. Flat routing, however, uses resident metadata linear in maximum `N`, and these controls keep experts resident; they do not satisfy the intended scalable resident-memory / hard external-parameter-probe contract.

### G3h — page granularity — 16 KiB SWEET SPOT IN THIS SETUP

At fixed 1 MiB external capacity with one probe and a shared fixed hash:

- 4 KiB / 256 pages / 960 active page MACs: `2.46392645`;
- 16 KiB / 64 pages / 3,936 MACs: `2.46253466`;
- 64 KiB / 16 pages / 16,128 MACs: `2.46337525`.

The 16 KiB condition is best in all three seeds. More bytes/probe are not automatically better; page/operator granularity is a real co-design variable.

See `docs/g3d_j_tinyshakespeare_controls.md` for full seeds, artifacts, and run identifiers.

### G4 — physical storage — BACKEND IMPLEMENTED, DEVICE STUDY PENDING

Linux `O_DIRECT + preadv` support exists. Publication-grade G4 requires named devices, repeated trials, queue-depth control, CPU measurements, end-to-end operator timing, and serialized/file-backed execution of trained LM pages.

## 8. Implementation policy

- Use reference implementations where closed-form structure isolates an invariant.
- Use PyTorch for learned routing/neural operators.
- Use `pread` for exact logical probe accounting and `O_DIRECT + preadv` where supported for physical-I/O experiments.
- Training may keep page tables resident, but inference claims require a separately serializable selected-page execution path.
- Report external bytes, block bytes, selected pages, logical bytes read, resident routing metadata, routing compute, active operator compute, and I/O mode.
- Retain negative results.
- The Tiny Shakespeare stage is now **frozen**; do not tune its learned-router boundary post hoc.

## 9. Current publication boundary

The strongest current empirical statement is:

> On canonical Tiny Shakespeare, inactive external page capacity can improve validation loss under a strict one-page external-parameter traffic budget and fixed active page compute. A paired fixed router scales monotonically through 256 pages. The present learned factorized router is strong through 64 pages but plateaus at 256, and conventional finite flat MoE routing remains lower-loss at `N=64`.

This is **not yet Q1-ready evidence**. The remaining blockers are concrete: reproduce the capacity effect on at least a second named language corpus; execute trained LM pages through the serialized/file-backed storage path under the same measured probe contract; and demonstrate the effect on a larger or more standard language-model setting before elevating the result from a controlled diagnostic to a publication claim.