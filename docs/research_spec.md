# ParamProbe research specification v0.7

## 1. Research question

Can learned parameter capacity increase on external storage while **worst-case external parameter traffic, resident working memory, and active inference compute remain bounded**, and does that extra inactive capacity improve learned task quality?

The central empirical hypothesis is:

> At fixed resident memory `M`, explicit external parameter traffic `Q`, and active compute `C`, task loss can improve as sparsely addressable external learned capacity `P_ext` increases.

The resource theorems define what is possible for the ParamProbe function family. They do **not** guarantee the empirical scaling hypothesis or finite-`N` superiority over conventional sparse models.

## 2. Parameter-probe model

Partition external learned state into `N` equal blocks `P_0, ..., P_{N-1}`, each exactly `B` bytes. For an input `x`, let `Pi_B(f, x)` be the number of external parameter-block reads used while evaluating the external-memory component of `f`.

A `q`-probe layer must satisfy

`sup_x Pi_B(f, x) <= q`.

Therefore logical explicit parameter traffic is bounded by

`Q <= q B`

per invocation, independent of total external capacity `N B`.

Physical device traffic is measured separately. Logical probe counting is not a substitute for uncached/direct-I/O measurements.

## 3. Page-sized conditional operator

Given resident hidden state `h in R^d`, a resident router returns at most `q` external block ids. Each selected block encodes a complete micro-operator `E_p`, for example a small MLP. A generic layer is

`y = h + sum_{p in R(h)} g_p(h) E_p(h)`.

For strict probe-bound experiments, each `E_p` must fit inside one block. Selected pages are streamed one at a time through a reusable `B`-byte workspace, so external-operator workspace does not scale with `N`.

`B` is an architectural hyperparameter rather than merely a filesystem property: larger pages provide greater local operator capacity and can amortize fixed storage latency, but increase bytes per probe and may reduce the number of addressable operators at fixed total storage.

## 4. Scalable addressing

A flat `N`-way router is excluded from the asymptotic claim because it can hide `Theta(N)` resident metadata.

For the reference factorized construction, let `N = m^r`. Represent an address as an `r`-tuple over radix `m`. With subkey width `s`, resident key metadata is

`Theta(r m s) = Theta(m s log_m N)`.

The complete Cartesian address space is never materialized. The exact top-k reference implementation uses best-first search and is tested against brute force on small spaces.

For empirical sweeps claiming **strictly fixed** resident router size and active routing compute, the maximum address width is allocated once and smaller capacities use prefixes/masks of that same output.

## 5. Core theoretical claims

### C1 — hard logical probe bound

If at most `q` complete block-sized operators are selected and evaluated sequentially, logical external parameter traffic is at most `qB`, independent of `N`.

### C2 — bounded external-operator workspace

Sequential page evaluation requires additional working storage of approximately

`B + O(d + q)`

up to implementation constants, independent of total external page count.

### C3 — sublinear routing metadata

For `N=m^r`, factor-key metadata is `Theta(rms)`. With constant `m,s`, routing metadata grows as `Theta(log N)` while external learned capacity grows as `Theta(NB)`.

### C4 — dense-function contrast

Known cell-probe lower bounds for exact online Boolean / `F_2` matrix-vector multiplication show that arbitrary dense transformations do not generally admit constant external probes with tiny auxiliary state. This is a contrast for a hard dense function family, not a claim that every conventional neural layer literally reads every stored weight on every input.

### C5 — routing reliability is a separate scaling resource

If an uncoded binary factor is correct independently with probability `p`, an `r`-bit address succeeds with probability `p^r`. Since `r=log_2 N`, address reliability can fall as the address space grows even while marginal factor accuracy appears high.

Error-correcting addresses can improve this under explicit channel assumptions, but negative code-rate results are retained. Coding is therefore a conditional sub-direction.

### C6 — composite-address collision can be regularized in `O(log N)` routing space

For factorized Bernoulli page probabilities `p_i` and `p_j`, the probability that two independently sampled complete addresses match is

`K(i,j) = product_f [p_if p_jf + (1-p_if)(1-p_jf)]`.

Averaging this estimates full-address collision probability without materializing the `N=2^r` address distribution. The normalized Rényi-2 deficit can therefore regularize composite page collapse using only factor probabilities. It does not tell the router which page is useful.

## 6. Non-claims

ParamProbe does not claim to invent model offloading, sparse Mixture-of-Experts, Product-Key routing, learned sparse memory, SSD-resident parameters, direct-I/O expert storage, error-correcting output codes, counterfactual router training, or collision/Rényi load balancing.

It also does **not** currently claim finite-`N` language-model quality superiority over conventional flat MoE routing. The Tiny Shakespeare controls show the opposite at `N=64`: a task-trained flat router is lower-loss than the present factorized learned router, even with an exact page-shape match and lower active matrix compute. The intended contribution is the **hard external parameter-probe budget as an architectural/scaling constraint**, together with scalable addressing, operator granularity, training, and physical storage.

## 7. Falsification gates

### G0 — invariants — PASSED

Reference tests establish exact factorized top-k retrieval on tested small spaces, file-backed/resident operator agreement, exact probe accounting, and exact `qB` logical bytes read. The pinned CPU CI currently passes all 10 repository tests.

### G1 — associative capacity — PASSED WITH ROUTING LIMITATIONS

A stateless one-probe associative memory matches an analytic collision law while external capacity grows at fixed 4096 bytes/query. Learned routing exposes an address-reliability wall; coding can help under some conditions but is not universally beneficial.

### G2 — synthetic nonlinear operators and task-only routing — PASSED WITH NEGATIVE ROUTING SUBCASES

The synthetic sequence establishes page-sized nonlinear expressivity, joint trainability under semantic controls, composite-address balancing, and a fixed-budget task-only counterfactual routing gate. Negative local-routing and coding results are retained where monotonicity fails.

### G3a — language-model insertion precheck — PASSED AS DIAGNOSTIC

The successful controlled insertion is

`embedding -> Transformer block 1 -> ParamProbe -> Transformer block 2 -> LM head`.

The page MLP is `48 -> 10 -> 48`, containing 1,018 FP32 parameters / 4,072 learned bytes in one 4,096-byte page and 960 active page matrix MACs/token. The environment-local source-code corpus precheck was monotone but is not a publication benchmark.

### G3b — learned causal routing architectural precheck — PASSED THROUGH 64 PAGES

A prefix-balanced, reliability-ordered causal router trained only from hidden states passes the local 1/4/16/64 architectural gate across three page-training seeds. No semantic address labels, realized next-token page targets, validation loss, or future-token information are used to train/order the router.

### G3c — canonical Tiny Shakespeare benchmark — MIXED / CONSTRAINING

The canonical Tiny Shakespeare run uses the verified Karpathy corpus (`1,115,394` bytes; Git blob SHA-1 `7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9`) and a fresh frozen byte-level backbone.

Under `q=1`, `B=4096`, the fixed hash is strictly monotone through 256 pages in all three seeds. The learned router improves through 64 pages but is slightly non-monotone at 256 in two of three seeds.

### G3f — paired Tiny Shakespeare capacity replication — CONFIRMS G3c BOUNDARY

A stricter replication explicitly pairs page-prefix initialization and training/evaluation minibatches across `N`.

Fixed hash three-seed means:

- 1 page: `2.46687950`;
- 4 pages: `2.46569404`;
- 16 pages: `2.46455893`;
- 64 pages: `2.46409338`;
- 256 pages: `2.46395130`.

The fixed-hash curve is strictly monotone in every seed.

Learned router means:

- 1 page: `2.46687950`;
- 4 pages: `2.46534705`;
- 16 pages: `2.46422635`;
- 64 pages: `2.46244999`;
- 256 pages: `2.46245038`.

Two of three seeds regress at `64 -> 256`; the mean is effectively a plateau. The eight-factor address-reliability boundary is therefore retained.

### G3g — compute-matched dense control — PARAMPROBE LOWER LOSS

A resident `48 -> 47 -> 48` adapter uses 4,512 matrix MACs/token, versus 4,544 for the learned ParamProbe router plus active page, but reaches `2.46482733` mean CE. Paired learned ParamProbe at 64 pages reaches `2.46244999`.

This supports the claim that the 64-page gain is not explained merely by comparable extra active dense compute.

### G3i/G3j — conventional flat MoE controls — PARAMPROBE DOES NOT WIN FINITE-N QUALITY

At maximum `N=64`, a flat task-trained router is a strong control.

G3i matches total matrix compute within 32 MACs/token using a `48 -> 15 -> 48` expert and reaches `2.46123072` mean CE at 64 experts, lower than learned ParamProbe.

G3j is stricter on operator shape: it uses the exact `48 -> 10 -> 48`, 4,072-byte expert and total inference matrix work of only 4,032 MACs/token, 512 fewer than learned ParamProbe. It reaches `2.46182543` mean CE at 64 experts, still lower than learned ParamProbe's `2.46244999`.

This is a genuine constraining result. The current factorized learned router is not better than a conventional flat router at finite `N=64`. Flat routing, however, uses resident metadata linear in maximum `N`, and these controls keep experts resident; they do not satisfy the intended scalable resident-memory / hard external-parameter-probe contract.

### G3h — paired page granularity — 16 KiB SWEET SPOT IN THIS SETUP

At fixed 1 MiB external capacity with one probe and a shared fixed hash:

- 4 KiB / 256 pages / 960 active page MACs: `2.46392645` mean CE;
- 16 KiB / 64 pages / 3,936 MACs: `2.46253466`;
- 64 KiB / 16 pages / 16,128 MACs: `2.46337525`.

The 16 KiB condition is best in all three seeds. More bytes/probe are therefore not automatically better; page/operator granularity is a real co-design variable.

See `docs/g3d_j_tinyshakespeare_controls.md` for full tables, seeds, artifacts, and run identifiers.

### G4 — physical storage — BACKEND IMPLEMENTED, DEVICE STUDY PENDING

Linux `O_DIRECT + preadv` support provides a reusable page-aligned direct-I/O backend. Existing local measurements are not publication hardware claims because the underlying storage/controller is uncharacterized.

Publication-grade G4 requires named devices, repeated trials, queue-depth control, CPU measurements, end-to-end operator timing, and serialized/file-backed execution of trained LM pages.

## 8. Implementation policy

- Use NumPy/reference implementations where closed-form structure isolates an invariant.
- Use PyTorch for learned routing/neural operators.
- Use `pread` for exact logical probe accounting.
- Use `O_DIRECT + preadv` where supported for page-cache-bypassing physical-I/O experiments.
- Training may keep a complete page table resident for convenience, but inference claims require a separately serializable selected-page execution path.
- Every experiment must report the resources it claims to hold fixed: external bytes, block bytes, selected pages, logical bytes read, resident routing metadata, routing compute, active operator compute, and I/O mode.
- Negative results are retained when they constrain the hypothesis.
- Named-corpus results are frozen once controls reveal a boundary; do not tune Tiny Shakespeare until the boundary disappears post hoc.

## 9. Current publication boundary

The strongest current empirical statement is:

> On canonical Tiny Shakespeare, inactive external page capacity can improve validation loss under a strict one-page external-parameter traffic budget and fixed active page compute. A paired fixed router scales monotonically through 256 pages. The present learned factorized router is strong through 64 pages but plateaus at 256, and conventional finite flat MoE routing remains lower-loss at `N=64`.

This is **not yet Q1-ready evidence**. The remaining blockers are concrete rather than conceptual: reproduce the capacity effect on at least a second named language corpus; execute trained LM pages through the serialized/file-backed storage path under the same measured probe contract; and demonstrate the effect on a larger or more standard language-model setting before elevating the result from a controlled diagnostic to a publication claim.