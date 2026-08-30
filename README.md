# ParamProbe

**Neural scaling under a hard external-parameter I/O budget.**

ParamProbe studies neural architectures whose learned parameter capacity may grow on external storage while inference obeys an explicit worst-case parameter-probe budget.

The core abstraction is a parameter store split into fixed-size blocks of `B` bytes. A ParamProbe layer may read at most `q` blocks per invocation, so explicit external parameter traffic is bounded by `q * B`, independent of total external capacity `N * B`.

The empirical hypothesis is:

> At fixed resident memory, active inference compute, and external parameter I/O, increasing sparsely addressable external learned capacity can improve task quality.

The repository is organized around falsification gates rather than a predetermined architecture.

## Current status

### G0 — resource invariants: passed

- exact factorized top-k retrieval matches brute force on tested small spaces;
- file-backed and resident page operators agree numerically;
- a `q=2`, `B=4096` smoke test performs exactly 8,192 explicit parameter bytes per invocation independent of external store size;
- Linux `O_DIRECT + preadv` support provides a page-cache-bypassing backend for physical-I/O studies.

### G1 — associative capacity: passed, with an address-reliability limitation

At fixed one-page I/O, increasing external capacity reduces collision-limited associative error. A learned factorized router can also exploit extra capacity, but marginal factor errors compound as address width grows. Error-correcting addresses remain a conditional sub-direction rather than a universal fix.

### G2a/G2b — nonlinear page operators: passed

External pages contain genuine nonlinear operators, including complete two-layer micro-MLPs. With routing controlled, task error decreases as the number of inactive external operators grows while active page shape, inference MACs, `q`, and `B` stay fixed.

For the 8→32→8 page MLP, three-seed normalized error falls from about `0.998` at one page to `0.035` at 256 pages with one 4 KiB page active per query.

See [`docs/g2_operator_capacity.md`](docs/g2_operator_capacity.md).

### G2c — jointly learned router + pages with address supervision: passed

A fixed router and nonlinear pages are trained jointly. In a hard routing setting the router reaches about 87.7% exact-address accuracy, uses essentially the whole page space, and normalized error falls from about `0.997` to `0.323` as pages grow 1→256. This establishes joint trainability but still supplies semantic page-address targets.

### G2d — task-only top-2 routing: negative/constraining

Removing address labels exposes a real failure: ordinary task loss plus factor-level balancing can discover useful specialization at high capacity but produces a non-monotone capacity curve. Balanced routing factors do not imply balanced **composite** addresses.

### G2e — composite-address Rényi balancing: passed as a load-balancing result, not a routing solution

For a factorized binary address, the full-address collision probability can be computed from `r=log2(N)` factor probabilities without materializing an `N`-way router. The normalized objective

`log(N * C2) = log(N) - H2`

directly controls composite-address collapse.

On the adversarial task it improves utilization and gives three-seed NMSE approximately:

`0.987 → 1.030 → 0.920 → 0.279` for 1/4/16/64 pages.

The 4-page regression is intentionally retained: **load balancing does not tell the router which page is useful for a particular input.**

See [`docs/theory_composite_collision.md`](docs/theory_composite_collision.md).

### G2f — fixed-budget counterfactual utility routing: passed synthetic gate

G2f supplies no semantic address labels. During training only, four candidate pages are evaluated per example and their observed task losses are distilled into the factorized router. The candidate budget is fixed at four for every multi-page condition. Inference is still a hard one-page route:

- `q_infer = 1`;
- `B = 4096 bytes`;
- identical router size and routing MACs across the sweep;
- identical active page-MLP shape and inference MACs;
- no dead pages in the three-seed 64-page runs.

On the deliberately adversarial control where semantic context prototypes and teacher MLPs are sampled independently, three-seed normalized MSE is:

| External pages | Normalized MSE |
|---:|---:|
| 1 | `0.9856 ± 0.0048` |
| 4 | `0.9557 ± 0.0133` |
| 16 | `0.8258 ± 0.0088` |
| 64 | **`0.2242 ± 0.0567`** |

This is the first ParamProbe gate with jointly learned nonlinear pages, **task-only routing, no address labels, fixed counterfactual training budget, and monotone held-out improvement as external capacity grows while inference remains one 4 KiB parameter probe.**

Counterfactual routing itself is prior work; see [`docs/g2f_counterfactual_routing.md`](docs/g2f_counterfactual_routing.md) for the novelty boundary and limitations.

### G3a — internal language-model precheck: passed with a fixed diagnostic router

The first LM attempt placed ParamProbe directly before the LM head and failed the monotone capacity gate. Diagnostics also showed that realized next-token counterfactual losses contain privileged future information and are therefore an invalid direct routing target for a causal language model.

G3a then freezes a two-block byte-level Transformer and inserts ParamProbe **between block 1 and block 2**. A fixed balanced six-factor context hash is allocated once and reused across the whole capacity sweep, which removes learned-routing instability from this capacity/insertion test.

The strict page operator is `48 -> 10 -> 48`:

- 1,018 FP32 learned parameters;
- 4,072 learned bytes, padded to one 4,096-byte page;
- `q_infer = 1`;
- identical fixed router metadata/MACs across 1/4/16/64 pages;
- identical active page MLP and downstream Transformer compute.

On the environment-local 1.2 MiB code-corpus precheck, the frozen backbone has validation CE `2.302792`. Across three independent page-training seeds:

| External pages | Validation CE |
|---:|---:|
| 1 | `2.29944 ± 0.00081` |
| 4 | `2.29751 ± 0.00037` |
| 16 | `2.29629 ± 0.00014` |
| 64 | **`2.29521 ± 0.00028`** |

The ordering is monotone in every seed and no validation page is dead. This is a controlled **precheck**, not a benchmark result: the container used local Python/PyTorch source because the standard Tiny Shakespeare file could not be downloaded into the runtime. Publication-grade G3 must rerun on a named corpus.

See [`docs/g3a_language_precheck.md`](docs/g3a_language_precheck.md).

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
python experiments/g2c_joint_supervised_router.py
python experiments/g2d_task_only_top2.py
python experiments/g2e_composite_collision.py
python experiments/g2f_counterfactual_utility.py
python experiments/g3a_internal_fixed_hash_lm.py --retrain-backbone
```

See [`docs/research_spec.md`](docs/research_spec.md) for the formal model and claims. The next core gate is **G3b: learned causal routing at the successful internal insertion point, without using realized future-token oracle labels as router targets.**
