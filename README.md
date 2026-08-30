# ParamProbe

**Neural scaling under a hard external-parameter I/O budget.**

ParamProbe studies neural architectures whose learned parameter capacity may grow on external storage while inference obeys an explicit worst-case parameter-probe budget.

The core abstraction is a parameter store split into fixed-size blocks of `B` bytes. A ParamProbe layer may read at most `q` blocks per invocation, so explicit external parameter traffic is bounded by `q * B`, independent of total external capacity `N * B`.

The empirical scaling hypothesis is:

> At fixed resident memory, active compute, and external parameter I/O, increasing sparsely addressable external learned capacity can improve task quality.

The repository is organized to falsify that hypothesis cheaply before language-model experiments.

## Current status

### G0 — resource invariants: passed

- exact factorized top-k retrieval matches brute force on tested small spaces;
- file-backed and resident page operators agree numerically;
- a `q=2`, `B=4096` smoke test performs exactly 8,192 explicit parameter bytes per invocation independent of external store size;
- Linux `O_DIRECT + preadv` support provides a page-cache-bypassing backend for later physical-I/O studies.

### G1 — associative capacity and routing: passed with a routing limitation

At fixed one-page I/O, increasing external capacity reduces collision-limited associative error. A fixed learned router also benefits from added capacity when address reliability is high enough.

The main failure mode is the **address-reliability wall**: a hard 12-factor router with about 92.5% marginal factor accuracy achieved only about 39.6% exact-address accuracy. Error-correcting parameter addresses improve this in a matched-compute proof-of-concept, but coding is retained as a conditional sub-direction rather than assumed to solve routing universally.

### G2a — nonlinear operator capacity: passed

Each external 4 KiB page stores a `64 x 16` FP32 matrix used in the nonlinear function

`f_i(z) = A_i tanh(R z + b)`.

The router, resident nonlinear feature map, active external matrix multiply, `q=1`, and `B=4096` are fixed. With a reliable router, increasing usable pages from 1 to 4096 reduces normalized operator error from about 1 to 0, closely matching the analytic collision law `1 - occupied/items`.

A deliberately noisy-router control leaves the oracle capacity curve intact but prevents the learned system from using the extra pages, cleanly separating operator capacity from routing reliability.

### G2b — fully learned page-sized MLPs: passed

Each external page contains a complete two-layer tanh micro-MLP. The default 8→32→8 operator has 552 FP32 parameters (2,208 learned bytes) and is padded/read as one physical 4 KiB parameter page.

Across three seeds, with oracle routing used to isolate operator expressivity, normalized held-out error is:

| External pages | Normalized MSE |
|---:|---:|
| 1 | 0.9980 ± 0.0005 |
| 4 | 0.9878 ± 0.0006 |
| 16 | 0.9519 ± 0.0029 |
| 64 | 0.8318 ± 0.0097 |
| 256 | **0.0349 ± 0.0004** |

Active MLP shape, active MACs, `q=1`, and 4 KiB parameter traffic/query are identical at every point; only inactive external operator count grows.

See [`docs/g2_operator_capacity.md`](docs/g2_operator_capacity.md) for the derivation, negative controls, and methodology.

## Reproduce

```bash
python -m pip install -e '.[dev,train]'
pytest -q
python experiments/g0_probe_invariants.py
python experiments/g1_hash_capacity.py
python experiments/g1b_learned_prefix_capacity.py
python experiments/g1c_coded_addressing.py
python experiments/g2a_nonlinear_basis_capacity.py
python experiments/g2b_micro_mlp_capacity.py
```

See [`docs/research_spec.md`](docs/research_spec.md) for the formal model, claims, non-claims, and experimental gates.
