# ParamProbe research specification v0.1

## 1. Research question

Can learned parameter capacity increase on external storage while **worst-case external parameter traffic, resident working memory, and active compute remain bounded**, and does that extra inactive capacity improve learning quality?

The central empirical hypothesis is:

> At fixed resident memory `M`, explicit external parameter traffic `Q`, and active compute `C`, task loss can improve as sparsely addressable external learned capacity `P_ext` increases.

The theory guarantees resource bounds for the proposed function family. It does **not** guarantee the empirical scaling hypothesis.

## 2. Parameter-probe model

Partition external learned state into `N` equal blocks `P_0, ..., P_{N-1}`, each exactly `B` bytes. For an input `x`, let `Pi_B(f, x)` be the number of block-read operations used to evaluate the external-memory part of `f`.

A `q`-probe layer must satisfy the hard invariant

`sup_x Pi_B(f, x) <= q`.

Logical explicit parameter traffic is therefore bounded by `q B` bytes per layer invocation. G4 will separately measure physical device traffic; logical reads are not a substitute for cold-storage measurements.

## 3. Page-sized conditional operator

Given resident hidden state `h in R^d`, a resident router returns at most `q` block ids. Each selected block encodes a complete micro-operator `E_p`, for example a low-rank MLP. The layer is

`y = h + sum_{p in R(h)} g_p(h) E_p(h)`.

The block size is an architectural hyperparameter. The implementation must never require an expert object larger than one block for the strict probe-bound experiments.

Blocks are streamed sequentially through a reusable `B`-byte workspace, so the external-operator workspace does not scale with `N`.

## 4. Scalable addressing

A flat `N`-way router is disallowed in the asymptotic claim because it can hide `Theta(N)` resident metadata.

For the first construction, let `N = m^r`. Represent each address as an `r`-tuple over radix `m`. With additive factor scores and factor-key width `s`, key metadata is `Theta(r m s) = Theta(m s log_m N)` scalars. The complete Cartesian address space is never materialized.

The current exact top-k implementation uses best-first search over the sorted additive-score lattice. G0 tests it against brute-force enumeration on small address spaces.

## 5. Claims we intend to establish

### C1 — hard logical probe bound

For a layer that selects at most `q` complete block-sized operators and evaluates them one at a time, logical external parameter traffic is at most `q B`, independent of `N`.

### C2 — bounded external-operator workspace

The additional working set required to evaluate selected external operators is `B + O(d + q)` bytes/scalars up to implementation constants, independent of `N`, provided blocks are processed sequentially and no external cache is counted as resident model state.

### C3 — sublinear routing metadata for the factorized construction

For `N=m^r`, resident factor-key metadata is `Theta(r m s)`. With constant `m,s`, it grows as `Theta(log N)` while external learned capacity grows as `Theta(NB)`.

### C4 — dense-function contrast, not an equivalence theorem

We will use known cell-probe/data-structure lower bounds for exact online matrix-vector multiplication as a contrast showing that arbitrary dense transformations do not generally admit constant external probes with tiny auxiliary state. We will **not** claim that every conventional dense neural layer literally requires reading every stored weight on every input.

A precise reduction and statement with assumptions belongs in a later theory note before publication claims are made.

## 6. Non-claims

ParamProbe does not claim to invent:

- model offloading or layer streaming;
- sparse Mixture-of-Experts;
- Product-Key routing;
- learned sparse memory layers;
- cache-aware expert routing;
- SSD/flash-resident model parameters;
- aligned/direct-I/O expert storage.

The intended novelty is the **hard external parameter-probe budget as an architectural/scaling constraint**, together with co-design of operator granularity and addressing under that constraint.

## 7. Falsification gates

### G0 — invariants

Pass only if:

- exact top-k factorized retrieval matches brute force on exhaustive/random small cases;
- file-backed operator output matches the same resident operator bytes;
- logical probes equal selected block count exactly;
- bytes read equal `q B` exactly.

### G1 — synthetic associative capacity

Construct a learned addressing task with many latent associations. Hold controller size, active compute, `q`, and `B` fixed while increasing `N`.

Primary question: does held-out association error improve with larger external capacity without increasing active I/O?

Pre-register a null outcome as meaningful: if capacity does not improve across at least a 16x `N` sweep under matched optimization, the main scaling hypothesis is weakened.

### G2 — router stress test

Sweep factor depth `r`, radix `m`, and top-k `q`. Measure utilization entropy, collision/concentration, retrieval accuracy, routing latency, and gradient starvation.

### G3 — language modeling

Only after G1/G2. Compare against matched dense, flat sparse-memory, and PEER-like baselines under resident-parameter, active-FLOP, and **external-bytes/token** constraints.

Primary plot: validation loss versus total external learned capacity at fixed `M`, `C`, and `Q`.

### G4 — physical storage

Add aligned uncached/direct-I/O backends. Report cold and warm results separately. Do not use mmap RSS or OS page-cache behavior as evidence of bounded physical memory/I/O.

## 8. Immediate implementation policy

The first code intentionally uses NumPy and explicit `pread`, not PyTorch. This isolates the addressing/probe semantics from autograd and framework caching. Training code enters only at G1.

Every experiment must emit the exact resource quantities it claims to hold fixed: total external bytes, block bytes, selected blocks, logical bytes read, and resident model metadata.
