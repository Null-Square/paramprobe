# Parameter-probe complexity as a neural specialization of cell-probe / external-memory models

Status: **FOUNDATIONAL NOTE / NOVELTY-NARROWING AUDIT**

## Executive conclusion

ParamProbe should **not** claim to invent the general idea of counting memory probes. The classical cell-probe model already charges memory-cell accesses while allowing arbitrary free computation, and the external-memory model adds a free resident cache plus block/page transfers.

The defensible contribution is a specialization for learned neural state:

> treat external learned parameters as a static data structure queried by each neural invocation, then report worst-case learned-parameter block probes, learned bytes transferred, resident learned/index state, reusable external-operator workspace, and active neural compute as separate resources.

This framing connects ParamProbe to established data-structure lower-bound theory rather than presenting an isolated bespoke complexity model.

## Prior theoretical foundation

### Cell-probe model

Classical references include Yao's 1981 table-search work and the later cell-probe literature of Fredman/Saks, Miltersen, Pătraşcu, Larsen, and others.

In the cell-probe model:

- a preprocessed data structure is stored in addressable cells of `w` bits;
- a query adaptively reads a sequence of cells;
- arbitrary computation on the query and previously read cell contents is free;
- query complexity is the number of probed cells.

A standard external-memory variant additionally gives the query algorithm a free resident cache and views external cells as pages/blocks. This is already very close to the systems abstraction motivating ParamProbe.

Primary background pointers:

- Andrew C.-C. Yao, *Should Tables Be Sorted?*, JACM 1981.
- Mihai Pătraşcu, *Logarithmic Lower Bounds in the Cell-Probe Model*, SIAM J. Comput. 2006 / related thesis material.
- External-memory/cached cell-probe discussions in the data-structure lower-bound literature.

### Static online matrix-vector multiplication lower bound

Raphaël Clifford, Allan Grønlund, and Kasper Green Larsen, *New Unconditional Hardness Results for Dynamic and Online Problems*, FOCS 2015, DOI `10.1109/FOCS.2015.71`, studies static online matrix-vector multiplication directly in the cell-probe model.

For an `n x n` matrix over a finite field, preprocessed into `S` cells of `w` bits, their theorem gives a nontrivial unconditional lower bound on the number of cell probes required to answer a matrix-vector query. In natural near-linear-space regimes the required query probes grow with `n`; constant-probe exact evaluation is not universal. The theorem is stronger than a hardware-specific bandwidth argument because computation between probes is free.

ParamProbe should cite this theorem rather than claim an original generic dense-function lower bound.

## ParamProbe specialization

Let one neural invocation receive activation/query `x`.

The learned state is partitioned into:

- resident state `R`, containing at most `M` bits/bytes of learned router/index/always-resident parameters;
- external learned store `Theta = (theta_1, ..., theta_N)`;
- each external block contains at most `b = 8B` bits (`B` bytes).

A parameter-probe inference algorithm may:

1. inspect `x` and resident state `R`;
2. choose an external learned block address;
3. read the complete block;
4. optionally choose later addresses based on earlier block contents (adaptive model);
5. after at most `q` reads, return the neural output.

Define

`Pi_B(A, x; Theta, R)`

as the number of external learned parameter blocks read for input `x`.

Worst-case parameter-probe complexity is

`Pi_B(A) = sup_x Pi_B(A, x; Theta, R)`.

The logical external learned-byte budget is

`Q_B(A) = B * Pi_B(A)`.

A `q`-probe architecture satisfies `Pi_B(A) <= q`, hence `Q_B(A) <= qB`, independent of total external learned capacity `NB`.

### Current ParamProbe experiments are a stricter non-adaptive special case

For the current `q=1` experiments, the external block address is determined entirely from the activation and resident router before payload access. Therefore the selected external block cannot influence its own address and there is no adaptive second-stage lookup.

This distinction should be reported explicitly because general cell-probe algorithms may be adaptive.

## Resources ParamProbe adds to ordinary cell-probe accounting

Cell-probe theory deliberately makes ordinary computation free. Neural systems cannot.

ParamProbe therefore reports a vector of resources rather than only probe count:

`(P_ext, Pi_B, Q, M, W, C)`

where:

- `P_ext = NB`: physical external learned storage capacity;
- `Pi_B`: worst-case independently addressed learned blocks/invocation;
- `Q <= Pi_B * B`: logical learned parameter bytes transferred;
- `M`: resident learned/router/index state;
- `W`: reusable working memory needed to stage/execute external blocks;
- `C`: active inference compute, with router and selected payload compute reported separately.

Physical device I/O, cache hits, queue depth, latency, and energy are measured systems quantities, not identified with logical `Q` by definition.

## Immediate propositions

### Proposition 1 — selected learned-byte bound

If every external learned block is at most `B` bytes and at most `q` blocks are read per invocation, logical external learned parameter traffic is at most `qB` bytes/invocation.

This is definitional, not a novelty claim.

### Proposition 2 — sequential external-operator workspace

If selected blocks are executed sequentially and a block can be decoded/executed using its own `B`-byte staging buffer plus `O(d+q)` activation/accumulator state, then external-operator workspace is

`W_ext <= B + O(d+q)`

rather than `qB` or `NB`.

The implementation must verify that no hidden resident copy of all selected/external parameters is materialized.

### Proposition 3 — fixed-radix factorized address scaling

Let `N = m^r`, where radix `m` and per-factor score width are constant. A router that independently scores `m` alternatives for each of `r` factors needs

- `Theta(r m s)` resident key/router state;
- `Theta(r m s)` score work;

for score dimension `s`.

Since `r = log_m N`, both are `Theta(log N)` for fixed `m,s`, while external learned capacity is `Theta(NB)`.

This is the address-scaling property tested by G8. It is not a claim of better routing quality.

### Proposition 4 — two-bank product-key address scaling

For exact two-bank product-key routing over `N` Cartesian addresses with fixed key dimension `d_key`, each bank contains `sqrt(N)` subkeys. Exact top-1 scoring therefore uses

`Theta(sqrt(N) * d_key)`

resident subkey state/work, plus the fixed activation-to-query projection.

G7b establishes that this can be the **better finite-N router** under the same `q=1` block contract; G8 establishes the exact address-resource crossover for the current dimensions.

## Borrowed non-universality result for dense learned functions

The most useful theoretical consequence comes from existing cell-probe lower bounds, not a new counting argument.

Consider an external learned object representing an arbitrary dense `n x n` matrix `M` over a finite field, and a neural-like query that must return `Mv` exactly for an online query vector `v`.

This is a static online matrix-vector data-structure problem. Clifford–Grønlund–Larsen (FOCS 2015) prove unconditional cell-probe lower bounds for this problem as a function of matrix dimension, field size, cell width, and representation space.

Because the cell-probe algorithm is allowed **free arbitrary computation**, its lower bound also applies to any more restrictive exact parameter-probe implementation with the same representation-space/cell-width regime.

Consequently:

> Constant external parameter probes with small auxiliary/resident state are not a universal replacement for arbitrary dense learned transformations at near-linear representation space.

This should replace any wording suggesting ParamProbe itself proves a novel general dense-function lower bound.

### Important caveats

The inherited lower bound is about a hard exact function family and a finite-field/static-data-structure setting. It does **not** imply:

- trained neural networks require dense parameter access;
- approximate real-valued inference has the same lower bound without further argument;
- sparse/conditional neural tasks cannot admit `q=O(1)`;
- exponentially larger precomputed external tables cannot trade space for probes;
- ParamProbe's empirical LM task lies in the hard matrix-vector family.

Its purpose is narrower: to establish a rigorous contrast showing that a bounded-probe architecture is a meaningful architectural restriction rather than a free universal representation trick.

## Consequence for the paper thesis

The theory section should say approximately:

> We adopt a neural specialization of classical cell-probe/external-memory accounting. The database is learned external parameter state and the query is a neural activation/token. Unlike the classical cell-probe model, we additionally charge resident learned state, active neural compute, selected learned bytes, workspace, and physical I/O. This specialization lets us ask an empirical scaling question: within a fixed probe/byte/compute envelope, can task quality improve as total external learned capacity increases?

That is defensible and connects the work to mature theory.

The paper should **not** say:

> We introduce the first model that counts parameter/memory probes.

## Open theory questions worth pursuing

1. **Approximate neural lower bounds.** Can known cell-probe / communication-complexity lower bounds be adapted to approximate real-valued matrix-vector or nonlinear transformations under a bounded error criterion relevant to neural inference?
2. **Resident-advice tradeoff.** Formalize the tradeoff between resident learned index bits `M` and external parameter probes `q` for conditional operator selection.
3. **Payload expressivity under one block.** Characterize function classes achievable when each fetched `B`-byte block encodes a complete nonlinear operator versus a vector/KV payload.
4. **Training versus query complexity.** The cell-probe model is an inference/query model; G6 empirically shows that training exposure can scale separately. A theory of learned preprocessing/training cost is outside the current claim.
5. **Physical block realization.** Logical cell probes are not identical to SSD I/O. Publication systems work must quantify index reads, filesystem/device amplification, cache behavior, and queueing separately.

## References

- A. C.-C. Yao. *Should Tables Be Sorted?* Journal of the ACM 28(3), 1981.
- M. Pătraşcu and E. D. Demaine. *Logarithmic Lower Bounds in the Cell-Probe Model.* SIAM Journal on Computing 35(4), 2006.
- R. Clifford, A. Grønlund, K. G. Larsen. *New Unconditional Hardness Results for Dynamic and Online Problems.* FOCS 2015. DOI: `10.1109/FOCS.2015.71`; arXiv: `1504.01836`.
- D. Chakraborty, L. Kamma, K. G. Larsen. *Tight Cell Probe Bounds for Succinct Boolean Matrix-Vector Multiplication.* arXiv: `1711.04467`.
