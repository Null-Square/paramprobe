# ParamProbe research specification v0.6

## 1. Research question

Can learned parameter capacity increase on external storage while **worst-case external parameter traffic, resident working memory, and active inference compute remain bounded**, and does that extra inactive capacity improve learned task quality?

The central empirical hypothesis is:

> At fixed resident memory `M`, explicit external parameter traffic `Q`, and active compute `C`, task loss can improve as sparsely addressable external learned capacity `P_ext` increases.

The resource theorems below define what is possible for the ParamProbe function family. They do **not** guarantee the empirical scaling hypothesis.

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

`B` is an architectural hyperparameter rather than merely a filesystem property: larger pages provide greater local operator capacity and can amortize fixed storage latency, but increase bytes per probe.

## 4. Scalable addressing

A flat `N`-way router is excluded from the asymptotic claim because it can hide `Theta(N)` resident metadata.

For the reference factorized construction, let

`N = m^r`.

Represent an address as an `r`-tuple over radix `m`. With subkey width `s`, resident key metadata is

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

Known cell-probe lower bounds for exact online Boolean / `F_2` matrix-vector multiplication show that arbitrary dense transformations do not generally admit constant external probes with tiny auxiliary state. Chakraborty, Kamma, and Larsen (STOC 2018) give `t r = ~Omega(n^3)` for `n < r < n^2` and `t = ~Omega(n^2)` for `r <= n` in their model.

This is a contrast for a hard dense function family. ParamProbe does **not** claim that every conventional neural layer literally reads every stored weight on every input.

### C5 — routing reliability is a separate scaling resource

If an uncoded binary factor is correct independently with probability `p`, an `r`-bit address succeeds with probability `p^r`. Since `r=log_2 N`, address reliability can fall as the address space grows even while marginal factor accuracy appears high.

Error-correcting addresses can improve this under explicit channel assumptions, but the repository retains negative code-rate results showing that redundancy can be difficult to learn. Coding is therefore a conditional sub-direction.

### C6 — composite-address collision can be regularized in `O(log N)` routing space

For factorized Bernoulli page probabilities `p_i` and `p_j`, the probability that two independently sampled complete addresses match is

`K(i,j) = product_f [p_if p_jf + (1-p_if)(1-p_jf)]`.

Averaging this quantity estimates full-address collision probability without materializing the `N=2^r` address distribution. The normalized Rényi-2 deficit

`Delta_2 = log(N C_2) = log N - H_2`

is zero for a uniform composite-address distribution and can be computed from only `r=O(log N)` factor probabilities per example.

This controls page collapse; it does not by itself tell the router which page is useful.

## 6. Non-claims

ParamProbe does not claim to invent:

- model offloading or layer streaming;
- sparse Mixture-of-Experts;
- Product-Key routing;
- learned sparse memory layers;
- cache-aware or stable expert routing;
- SSD/flash-resident model parameters;
- aligned/direct-I/O expert storage;
- Error-Correcting Output Codes;
- counterfactual or preference-based router training;
- collision/Rényi load balancing in flat MoE systems.

The intended contribution is the **hard external parameter-probe budget as an architectural/scaling constraint**, together with co-design of operator granularity, scalable addressing, routing, training, and physical storage.

## 7. Falsification gates

### G0 — invariants — PASSED

Reference tests establish exact factorized top-k retrieval on tested small spaces, file-backed/resident operator agreement, exact probe accounting, and exact `qB` logical bytes read.

A 16 MiB store with `B=4096`, `q=2`, and 200 randomized trials produced exactly 8192 logical bytes/invocation and zero numerical disagreement from resident execution.

### G1 — associative capacity — PASSED WITH ROUTING LIMITATIONS

A stateless one-probe associative memory matches an analytic collision law while external capacity grows at fixed 4096 bytes/query. A learned fixed-width router can also use additional pages when routing is sufficiently reliable.

G1b exposes the address-reliability wall. G1c/G1d show that error-correcting addresses can help but are not universally beneficial.

### G2a — analytically controlled nonlinear operators — PASSED

For `f_i(z)=A_i tanh(Rz+b)`, a reliable fixed router gives a capacity curve that closely matches the analytic collision law while one 4 KiB page is active per query.

### G2b — fully learned page MLP expressivity — PASSED

The reference 8→32→8 page MLP contains 552 FP32 learned parameters (2208 bytes) inside one 4096-byte page. With oracle routing used only to isolate page expressivity, three-seed held-out NMSE falls from approximately `0.998` at one page to `0.035` at 256 pages with identical active operator shape/MACs and one page probe.

### G2c — joint router/pages with semantic address supervision — PASSED AS A CONTROL

With supervised semantic addresses, router and nonlinear pages train jointly and use the available page space without collapse. This establishes joint trainability but is not a task-only routing result.

### G2d — task-only local routing — NEGATIVE / CONSTRAINING

Task-only top-2-neighborhood training can discover useful specialization at high capacity but gives a non-monotone capacity curve. Factor-level balance does not guarantee composite-address utilization.

### G2e — composite-address Rényi balancing — PASSED AS LOAD-BALANCING RESULT

Full-address collision/Rényi regularization substantially improves composite page utilization without `O(N)` router state, but the adversarial task retains a small-capacity regression. Load balancing solves collapse, not routing utility.

### G2f — fixed-budget counterfactual utility routing — PASSED SYNTHETIC GATE

Training evaluates a fixed four candidate pages per example and distills their task utility into the factorized router. No semantic address labels are used. The candidate budget is fixed across all multi-page conditions, while inference remains exactly one 4 KiB page.

On an adversarial synthetic task with independent semantic contexts and teacher functions, three-seed NMSE is approximately:

- 1 page: `0.9856 +/- 0.0048`;
- 4 pages: `0.9557 +/- 0.0133`;
- 16 pages: `0.8258 +/- 0.0088`;
- 64 pages: `0.2242 +/- 0.0567`.

This is the first task-only, no-address-label synthetic result with monotone improvement as inactive external capacity grows under a fixed inference probe budget.

See `docs/g2f_counterfactual_routing.md`.

### G3a — language-model capacity/insertion precheck — PASSED WITH FIXED DIAGNOSTIC ROUTER

The first LM attempt attached ParamProbe directly before the output head and **failed** monotone scaling. Diagnostics revealed two language-specific issues:

1. realized next-token counterfactual losses contain privileged future information, so the realized best page is not a valid direct causal routing target;
2. an expected-utility critic can remove that leakage but may collapse page functions into redundant adapters.

A stable fixed context hash was then used to separate routing from insertion depth. The pre-head insertion still failed monotone scaling.

The successful controlled architecture is:

`embedding -> Transformer block 1 -> ParamProbe -> Transformer block 2 -> LM head`.

The frozen backbone, maximum six-factor hash metadata, routing MACs, active page operator, downstream compute, training schedule, `q=1`, and `B=4096` are fixed across the page-count sweep.

The strict page MLP is `48 -> 10 -> 48`:

- 1,018 FP32 learned parameters;
- 4,072 learned bytes padded to one 4,096-byte page;
- 960 active page matrix MACs/token.

On the environment-local 1.2 MiB code-corpus precheck, frozen-backbone validation CE is `2.3027921`. Across three independent page-training seeds:

- 1 page: `2.29944 +/- 0.00081`;
- 4 pages: `2.29751 +/- 0.00037`;
- 16 pages: `2.29629 +/- 0.00014`;
- 64 pages: `2.29521 +/- 0.00028`.

The ordering is monotone in every seed and there are no dead validation pages.

This is **not a benchmark claim** because the execution environment used local source-code text after standard corpus download failed. See `docs/g3a_language_precheck.md`.

### G3b — learned causal routing architectural precheck — PASSED THROUGH 64 PAGES

G3b keeps the successful internal insertion and strict 4 KiB page operator while replacing the fixed hash with a causal learned factorized router.

The first strict shared-router reconstruction is retained as a negative result: it improved strongly at 16/64 pages but seed 9 had a small `1 -> 4` regression.

A predeclared follow-up then added:

1. equal composite Rényi-2 balance on the raw 2-, 4-, and 6-factor prefixes;
2. router freezing before page training;
3. a fixed factor order chosen only from clean-vs-perturbed hard-bit agreement on a deterministic training-hidden-state bank.

No semantic page labels, realized next-token counterfactual targets, validation loss, or future-token information are used to train or order the router.

On the same environment-local diagnostic corpus, three page-training seeds give:

- 1 page: `2.29944 +/- 0.00081`;
- 4 pages: `2.29802 +/- 0.00018`;
- 16 pages: `2.29423 +/- 0.00031`;
- 64 pages: `2.29165 +/- 0.00025`.

The ordering is strictly monotone in every seed. There are no dead validation pages. `q=1`, the page payload remains 4,072 learned bytes inside one 4,096-byte block, and logical external parameter traffic remains exactly 4,096 bytes/token.

This passes the **architectural learned-router precheck through 64 pages**, but it remains a local-corpus diagnostic rather than a named benchmark result.

See `docs/g3b_learned_causal_routing.md` and `docs/g3b_prefix_reliable_routing.md`.

### G3c — Tiny Shakespeare named-corpus capacity sweep — MIXED / CONSTRAINING

G3c freezes the G3 architecture/training protocol, switches to canonical Tiny Shakespeare, allocates an eight-factor maximum address width, and evaluates `N = 1, 4, 16, 64, 256` on a fresh common frozen backbone.

The canonical corpus is verified before training by Git blob SHA-1:

`7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9`.

The frozen-backbone validation CE is `2.47749685`.

#### Fixed-hash control

Three-seed mean validation CE is:

- 1 page: `2.46669 +/- 0.00092`;
- 4 pages: `2.46565 +/- 0.00023`;
- 16 pages: `2.46457 +/- 0.00015`;
- 64 pages: `2.46395 +/- 0.00009`;
- 256 pages: `2.46370 +/- 0.00014`.

The fixed-hash ordering is strictly monotone in **every seed through 256 pages** while `q=1`, `B=4096`, and active page compute remain fixed. At 256 pages, normalized hard-route entropy is about `0.721` and 9.375% of pages are dead on validation, so the capacity effect does not require perfect page utilization.

#### Learned prefix-balanced reliability-ordered router

Three-seed mean validation CE is:

- 1 page: `2.46669 +/- 0.00092`;
- 4 pages: `2.46544 +/- 0.00019`;
- 16 pages: `2.46427 +/- 0.00007`;
- 64 pages: **`2.46245 +/- 0.00024`**;
- 256 pages: `2.46251 +/- 0.00009`.

All three seeds improve monotonically through 64 pages. Seed 7 also improves at 256 pages, but seeds 8 and 9 regress slightly at `64 -> 256`, and the mean change is `+0.00005844` CE. Therefore the strict learned-router 256-page extension is **not passed**.

At 256 pages, learned-router utilization entropy remains about `0.924`, dead-page fraction is only `0.0039`, but full-address perturbation stability has fallen to about `0.817`. The coincidence is consistent with the earlier address-reliability warning but does not establish causality.

The learned router is lower-loss than the fixed hash at matched page count from 4 through 256 pages, but this is not a total-compute-matched comparison because the learned router uses more routing MACs.

G3c therefore supports a narrow named-corpus capacity claim while preserving a learned-routing scaling boundary:

> On Tiny Shakespeare, validation loss improves as external inactive page capacity grows at fixed one-page external traffic and fixed active page compute. A fixed hash remains monotone through 256 pages; the current learned causal router is robust through 64 pages and marginally non-monotone at 256 pages.

See `docs/g3c_tinyshakespeare_capacity.md` and `.github/workflows/g3c_tinyshakespeare.yml`.

### G4 — physical storage — BACKEND IMPLEMENTED, DEVICE STUDY PENDING

Linux `O_DIRECT + preadv` support provides a reusable page-aligned direct-I/O backend. The local container precheck shows a strong block-size/throughput tradeoff, but those measurements are not publication hardware claims because the underlying storage/controller is uncharacterized.

Publication-grade G4 requires named devices, repeated trials, queue-depth control, CPU measurements, and end-to-end operator timing.

## 8. Implementation policy

- Use NumPy/reference implementations where closed-form structure isolates an invariant.
- Use PyTorch for learned routing/neural operators.
- Use `pread` for exact logical probe accounting.
- Use `O_DIRECT + preadv` where supported for page-cache-bypassing physical-I/O experiments.
- Training may keep a complete page table resident for convenience, but inference claims require a separately serializable selected-page execution path.
- Every experiment must report the resources it claims to hold fixed: external bytes, block bytes, selected pages, logical bytes read, resident routing metadata, routing compute, active operator compute, and I/O mode.
- Negative results are retained when they constrain the hypothesis.

The next core milestone is **matched Tiny Shakespeare baselines and page-size sweeps**, while keeping the G3c `64 -> 256` learned-router regression frozen rather than tuning it away.
