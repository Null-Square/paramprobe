# ParamProbe

**Neural scaling under a hard external-parameter I/O budget.**

ParamProbe studies neural architectures whose learned parameter capacity may grow on external storage while inference obeys an explicit worst-case parameter-probe budget.

The core abstraction is a parameter store split into fixed-size blocks of `B` bytes. A ParamProbe layer may read at most `q` blocks per invocation, so explicit external parameter traffic is bounded by `q * B`, independent of total external capacity `N * B`.

This repository starts with three claims we can test before training any language model:

1. **Probe invariant:** file-backed execution performs at most `q` external block reads.
2. **Working-memory invariant:** streaming one selected block at a time does not require memory proportional to the external parameter count `N`.
3. **Addressability invariant:** a factorized router can represent an address space `N = m^r` using metadata that grows with `r = log_m N`, without enumerating all `N` addresses.

The empirical scaling hypothesis is intentionally separate from those invariants:

> At fixed resident memory, active compute, and external parameter I/O, increasing sparsely addressable external parameter capacity can improve learned task quality.

That hypothesis may fail. The repository is organized to falsify it cheaply before expensive language-model experiments.

## First milestone

Run the deterministic G0 checks:

```bash
python -m pip install -e '.[dev]'
pytest -q
python experiments/g0_probe_invariants.py
```

See [`docs/research_spec.md`](docs/research_spec.md) for the formal model, claims, non-claims, and experimental gates.
