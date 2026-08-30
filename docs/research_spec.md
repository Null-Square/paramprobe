# ParamProbe research specification v0.3

## 1. Research question

Can learned parameter capacity increase on external storage while **worst-case external parameter traffic, resident working memory, and active compute remain bounded**, and does that extra inactive capacity improve learning quality?

The central empirical hypothesis is:

> At fixed resident memory `M`, explicit external parameter traffic `Q`, and active compute `C`, task loss can improve as sparsely addressable external learned capacity `P_ext` increases.

The theory guarantees resource bounds for the proposed function family. It does **not** guarantee the empirical scaling hypothesis.

## 2. Parameter-probe model

Partition external learned state into `N` equal blocks `P_0, ..., P_{N-1}`, each exactly `B` bytes. For an input `x`, let `Pi_B(f, x)` be the number of block-read operations used to evaluate the external-memory part of `f`.

A `q`-probe layer must satisfy the hard invariant

`sup_x Pi_B(f, x) <= q`.

Logical explicit parameter traffic is therefore bounded by `q B` bytes per layer invocation. Physical device traffic is measured separately; logical reads are not a substitute for uncached storage measurements.

## 3. Page-sized conditional operator

Given resident hidden state `h in R^d`, a resident router returns at most `q` block ids. Each selected block encodes a complete micro-operator `E_p`, for example a low-rank MLP. The layer is

`y = h + sum_{p in R(h)} g_p(h) E_p(h)`.

The implementation must never require an expert object larger than one block for strict probe-bound experiments. Blocks are streamed sequentially through a reusable `B`-byte workspace, so the external-operator workspace does not scale with `N`.

`B` is an **architectural hyperparameter**, not merely a filesystem detail. Larger pages provide more local operator capacity and better amortize fixed storage-read latency, but increase bytes per probe. Experiments must therefore sweep `B` jointly with `q`.

## 4. Scalable addressing

A flat `N`-way router is disallowed in the asymptotic claim because it can hide `Theta(N)` resident metadata.

For the first asymptotic construction, let `N = m^r`. Represent each address as an `r`-tuple over radix `m`. With additive factor scores and factor-key width `s`, key metadata is

`Theta(r m s) = Theta(m s log_m N)`

scalars. The complete Cartesian address space is never materialized.

The exact top-k reference implementation uses best-first search over the sorted additive-score lattice and is tested against brute force on small spaces.

For empirical sweeps that claim **strictly fixed** resident router size and active routing compute, the router is allocated once at maximum address width. Smaller external-capacity conditions use prefixes/masks of the same output so controller parameters and routing compute remain literally fixed.

## 5. Claims we intend to establish

### C1 — hard logical probe bound

For a layer that selects at most `q` complete block-sized operators and evaluates them one at a time, logical external parameter traffic is at most `q B`, independent of `N`.

### C2 — bounded external-operator workspace

The additional working set required to evaluate selected external operators is `B + O(d + q)` bytes/scalars up to implementation constants, independent of `N`, provided blocks are processed sequentially and no external cache is counted as resident model state.

### C3 — sublinear routing metadata for the factorized construction

For `N=m^r`, resident factor-key metadata is `Theta(r m s)`. With constant `m,s`, it grows as `Theta(log N)` while external learned capacity grows as `Theta(NB)`.

### C4 — dense-function contrast, not an equivalence theorem

Known cell-probe lower bounds for exact online Boolean / `F_2` matrix-vector multiplication show that arbitrary dense transformations do not generally admit constant external probes with tiny auxiliary state. In particular, Chakraborty, Kamma, and Larsen (STOC 2018) show a tradeoff `t r = ~Omega(n^3)` for `n < r < n^2`, and `t = ~Omega(n^2)` for `r <= n`, where `r` is side information and `t` is query probes.

We will **not** claim that every conventional dense neural layer literally requires reading every stored weight on every input. The lower bound is a contrast for a hard dense function family.

### C5 — routing reliability is a separate scaling resource

G1b exposed that logarithmic routing metadata does not imply reliable addressing. Under an independent binary-factor error model with per-factor correctness `p`, a raw `r`-bit address succeeds with probability `p^r`. Since `r = log_2 N`, this tends to zero for fixed `p < 1`.

Error-correcting addresses can improve this finite reliability and, under explicit channel assumptions, can preserve asymptotic reliability with only `Theta(log N)` routing symbols. This is currently a **conditional sub-theory**, not the central ParamProbe claim. See `docs/theory_address_reliability.md` and `docs/theory_routing_channel.md`.

## 6. Non-claims

ParamProbe does not claim to invent:

- model offloading or layer streaming;
- sparse Mixture-of-Experts;
- Product-Key routing;
- learned sparse memory layers;
- cache-aware or stable expert routing;
- SSD/flash-resident model parameters;
- aligned/direct-I/O expert storage;
- Error-Correcting Output Codes.

The intended novelty is the **hard external parameter-probe budget as an architectural/scaling constraint**, together with co-design of operator granularity, routing, and physical storage under that constraint.

## 7. Falsification gates

### G0 — invariants — PASSED

Initial deterministic checks establish:

- exact factorized top-k retrieval matches brute force on tested small spaces;
- file-backed operator output matches resident operator bytes;
- logical probes equal selected block count exactly;
- bytes read equal `q B` exactly.

A 16 MiB file-backed store with `B=4096`, `q=2`, and 200 randomized trials produced exactly 8192 logical bytes/invocation and zero numerical difference from resident execution.

### G1a — collision-limited associative capacity — PASSED

A stateless deterministic one-probe router isolates the capacity/I-O abstraction. On 16,384 iid associations, increasing external capacity from 0.25 MiB to 1 GiB reduced MSE from about 0.996 to 0.031 while parameter traffic remained exactly 4096 bytes/query. The curve closely tracks the analytic collision floor.

This proves only that additional inactive external slots can carry useful independent state at fixed probe cost; it is not a neural-routing result.

### G1b — learned fixed-controller capacity — PASSED WITH A LIMITATION

A single fixed 12-factor router was trained once and reused across the whole capacity sweep.

Hard regime (`noise=0.12`):

- marginal factor accuracy: about 0.925;
- 12-bit exact-address accuracy: about 0.396;
- increasing page count did not materially improve MSE because routing errors dominated.

Easier regime (`noise=0.03`):

- marginal factor accuracy: about 0.972;
- exact-address accuracy: about 0.717;
- increasing external capacity from 64 to 4096 pages reduced learned-memory MSE from about 0.987 to 0.371 while controller parameters, routing MACs, `q`, and `B` stayed fixed.

Interpretation: the central capacity mechanism survives learned routing when routing is reliable enough, but useful scaling is limited by address stability.

### G1c — coded address reliability — PROMISING SUB-RESULT

A Hamming(15,11) page-address experiment compares raw and coded routers with nearly identical controller parameter/MAC budgets. At the default moderate-noise setting over three seeds:

- raw exact-address accuracy: about `0.591 +/- 0.004`;
- coded exact-address accuracy: about `0.838 +/- 0.005`;
- raw full-capacity associative MSE: about `0.747 +/- 0.005`;
- coded MSE: about `0.359 +/- 0.018`.

Both conditions issue one 4 KiB external parameter probe.

### G1d — code-rate precheck — NEGATIVE / CONSTRAINING

A matched-budget systematic-linear-code experiment showed that simply adding parity outputs can make the router's symbol-prediction task harder enough to erase the coding gain when the frozen semantic representation does not already support redundant evidence.

Therefore ParamProbe does **not** assume that error-correcting addresses are universally beneficial. Any coded-routing claim must include the representation-learning cost and error correlations.

### G2 — end-to-end sparse-operator scaling — NEXT CORE GATE

Move beyond page-value lookup to page-sized neural operators. Hold fixed:

- resident backbone/controller parameter count;
- active routing MACs;
- selected-page count `q`;
- page size `B` for each matched sweep;
- external bytes/query `qB`.

Increase only total external operator count `N` and test whether held-out task quality improves. Measure expert/page utilization, routing entropy, instability under perturbation, and gradient starvation.

G2 is the gate that decides whether ParamProbe is a neural architecture rather than only an associative-memory/data-structure result.

### G3 — language modeling

Only after G2. Compare against matched dense, flat sparse-memory, PEER-like, and conventional MoE baselines under resident-parameter, active-FLOP, and **external-bytes/token** constraints.

Primary plot: validation loss versus total external learned capacity at fixed `M`, `C`, and `Q`.

### G4 — physical storage — BACKEND IMPLEMENTED, DEVICE STUDY PENDING

An aligned Linux `O_DIRECT + preadv` backend now bypasses the normal OS page cache using a reusable page-aligned buffer.

A local 512 MiB precheck confirms the benchmark path and shows a strong block-granularity tradeoff. Approximate random-read results on this uncharacterized container storage were:

- 4 KiB: ~8.3k probes/s, ~32 MiB/s;
- 16 KiB: ~15.3k probes/s, ~240 MiB/s;
- 64 KiB: ~13.8k probes/s, ~865 MiB/s;
- 256 KiB: ~9.5k probes/s, ~2.37 GiB/s.

These are not hardware claims; device/controller caching is not controlled. Publication-grade G4 requires named devices, repeated trials, queue-depth control, CPU measurements, and end-to-end operator timing.

## 8. Immediate implementation policy

- G0/G1a remain NumPy/reference implementations.
- PyTorch is used for learned-routing and upcoming neural-operator experiments.
- `pread` is used for exact logical probe accounting.
- `O_DIRECT + preadv` is used where supported for page-cache-bypassing physical-I/O experiments.
- Every experiment must emit the resource quantities it claims to hold fixed: external bytes, block bytes, selected blocks, logical bytes read, controller parameters, routing compute, and relevant cache/I/O mode.
- Negative results are retained in the repository when they constrain the hypothesis.

The next core milestone is **G2 end-to-end page-sized neural operators under a fixed external parameter-probe budget**.
