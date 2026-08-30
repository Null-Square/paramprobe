# ParamProbe research specification v0.4

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

Given resident hidden state `h in R^d`, a resident router returns at most `q` block ids. Each selected block encodes a complete micro-operator `E_p`, for example a small MLP. A generic layer is

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

Error-correcting addresses can improve finite reliability and, under explicit channel assumptions, can preserve asymptotic reliability with only `Theta(log N)` routing symbols. This remains a **conditional sub-theory**, not the central ParamProbe claim. See `docs/theory_address_reliability.md` and `docs/theory_routing_channel.md`.

### C6 — nonlinear operator collision law for linear-in-page-parameter families

For operators of the form

`f_i(z) = A_i phi(z)`,

where `phi` may be nonlinear but is shared/resident and `A_i` is page-local, the population-risk-minimizing page parameter for a collision group is the arithmetic mean of the item-specific `A_i` matrices. For iid zero-mean page parameters, the expected normalized collision error is `1 - O/A`, where `A` is the number of semantic items and `O` is the number of occupied pages.

G2a verifies this law numerically with `phi(z)=tanh(Rz+b)` and a one-page, 4 KiB parameter budget.

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

The intended novelty is the **hard external parameter-probe budget as an architectural/scaling constraint**, together with co-design of operator granularity, routing, learning, and physical storage under that constraint.

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

### G1b — learned fixed-controller capacity — PASSED WITH A LIMITATION

A single fixed 12-factor router was trained once and reused across the whole capacity sweep. A hard regime exposed the address-reliability wall; an easier regime demonstrated capacity gains at literally fixed controller parameters, routing MACs, `q`, and `B`.

### G1c/G1d — coded addressing — PROMISING BUT CONDITIONAL

A matched-compute Hamming(15,11) experiment improved exact-address accuracy and associative MSE, but a systematic code-rate precheck showed that redundant parity outputs can become harder to learn and erase the gain. Coding is therefore not assumed to solve routing universally.

### G2a — analytically controlled nonlinear operators — PASSED

Each semantic item owns a nonlinear function

`f_i(z) = A_i tanh(R z + b)`.

The resident feature map, router architecture, active page multiply, `q=1`, and `B=4096` are fixed. One `64 x 16` FP32 matrix fills exactly one 4 KiB external parameter page.

With a reliable fixed router, increasing pages from 1 to 4096 reduces normalized operator MSE from about 1 to 0 and matches the analytic collision law closely across three seeds. A deliberately noisy-router control leaves the oracle curve intact while the learned system fails to exploit larger capacity, separating capacity from routing reliability.

See `docs/g2_operator_capacity.md`.

### G2b — fully learned page-sized nonlinear MLPs — PASSED

Each selected page contains a complete two-layer tanh micro-MLP. The reference 8→32→8 operator has 552 FP32 learned parameters (2208 payload bytes) inside one 4096-byte parameter page. The active operator performs the same 512 matrix MACs/query at every capacity point.

Across three seeds with oracle routing to isolate operator expressivity, normalized held-out MSE changes as:

- 1 page: `0.9980 +/- 0.0005`;
- 4 pages: `0.9878 +/- 0.0006`;
- 16 pages: `0.9519 +/- 0.0029`;
- 64 pages: `0.8318 +/- 0.0097`;
- 256 pages: `0.0349 +/- 0.0004`.

Only inactive external operator count changes. Active MLP shape, active compute, `q=1`, and parameter traffic of 4096 bytes/query remain fixed.

The page layout is serializable and file-backed inference is separately tested under the one-probe contract.

### G2c — jointly trained routing + external operators — NEXT CORE GATE

The next experiment must remove the remaining separation between routing and operator learning while retaining a fixed-resource comparison.

Requirements:

- router architecture/parameter count fixed across `N`;
- page MLP shape fixed across `N`;
- hard `q`-page inference path preserved;
- task loss and routing mechanism trained in the same run;
- report utilization entropy, dead-page fraction, page-update counts, routing stability under perturbation, and held-out task loss;
- include oracle-routing and frozen-router controls to attribute failure correctly.

A result counts as progress only if extra external pages improve held-out quality without hidden `O(N)` resident routing state or increased active parameter traffic.

### G3 — language modeling — BLOCKED ON G2c

Do not start language modeling merely because G2a/G2b pass. First demonstrate that routing and page-local operators can be learned together without collapse under the fixed probe/resource envelope.

If G2c passes, compare against matched dense, flat sparse-memory, PEER-like, and conventional MoE baselines under resident-parameter, active-FLOP, and **external-bytes/token** constraints.

Primary plot: validation loss versus total external learned capacity at fixed `M`, `C`, and `Q`.

### G4 — physical storage — BACKEND IMPLEMENTED, DEVICE STUDY PENDING

An aligned Linux `O_DIRECT + preadv` backend bypasses the normal OS page cache using a reusable page-aligned buffer.

A local 512 MiB precheck confirms the benchmark path and shows a strong block-granularity tradeoff. Approximate random-read results on this uncharacterized container storage were:

- 4 KiB: ~8.3k probes/s, ~32 MiB/s;
- 16 KiB: ~15.3k probes/s, ~240 MiB/s;
- 64 KiB: ~13.8k probes/s, ~865 MiB/s;
- 256 KiB: ~9.5k probes/s, ~2.37 GiB/s.

These are not hardware claims; device/controller caching is not controlled. Publication-grade G4 requires named devices, repeated trials, queue-depth control, CPU measurements, and end-to-end operator timing.

## 8. Immediate implementation policy

- NumPy/reference experiments are used where closed-form structure lets us isolate an invariant.
- PyTorch is used for learned routing and neural-operator experiments.
- `pread` is used for exact logical probe accounting.
- `O_DIRECT + preadv` is used where supported for page-cache-bypassing physical-I/O experiments.
- Training may keep the full page table resident for convenience, but inference/resource claims count only a selected-page execution path and must be separately serialized/validated.
- Every experiment must emit the resource quantities it claims to hold fixed: external bytes, block bytes, selected blocks, logical bytes read, controller parameters, routing compute, active operator compute, and relevant cache/I/O mode.
- Negative results are retained when they constrain the hypothesis.

The next core milestone is **G2c jointly trained routing and page-local operators under a fixed external parameter-probe budget**.
