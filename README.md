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
- a `q=2`, `B=4096` smoke test performs exactly 8,192 explicit parameter bytes per invocation independent of external store size.

### G1a — stateless capacity scaling: passed

With one 4 KiB parameter probe per query and zero router state, increasing external associative capacity reduces collision-limited error in close agreement with the analytic occupancy prediction.

### G1b — learned raw addresses: exposed a failure mode

A fixed 12-factor learned router achieved about 92.5% marginal factor accuracy in the hard setting, but only about 39.6% full-address accuracy. The observed value closely matches the simple independence prediction `0.925^12`.

This is the **address-reliability wall**: raw exponentially factorized addresses can become unreliable even when every individual routing decision looks accurate.

### G1c — error-correcting parameter addresses: promising

A matched-compute Hamming(15,11) proof-of-concept predicts a redundant codeword, syndrome-decodes it into a logical page address, and then performs the same single external parameter probe.

At the default moderate-noise setting across three seeds:

- raw full-address accuracy: about `0.591 +/- 0.004`;
- coded full-address accuracy: about `0.838 +/- 0.005`;
- raw full-capacity associative-memory MSE: about `0.747 +/- 0.005`;
- coded MSE: about `0.359 +/- 0.018`.

The controllers are closely budget matched: about 9.36k parameters and 9.23k routing MACs/query each. Both read exactly one 4 KiB external page.

See [`docs/theory_address_reliability.md`](docs/theory_address_reliability.md) for the emerging theory.

## Reproduce

```bash
python -m pip install -e '.[dev,train]'
pytest -q
python experiments/g0_probe_invariants.py
python experiments/g1_hash_capacity.py
python experiments/g1b_learned_prefix_capacity.py
python experiments/g1c_coded_addressing.py
```

See [`docs/research_spec.md`](docs/research_spec.md) for the formal model, claims, non-claims, and experimental gates.
