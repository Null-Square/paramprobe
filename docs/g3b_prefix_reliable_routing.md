# G3b — learned causal routing with prefix balance and reliability ordering

## Status

**PASSED as an architectural diagnostic gate on the environment-local corpus.**

This is still not a publication benchmark result. The corpus is the same deterministic
1.2 MiB local Python/PyTorch source fallback used by G3a. The next experiment should
move to a recognized small causal-language benchmark rather than further tuning this
local diagnostic.

## Why this refinement exists

The first reproducible shared learned-router reconstruction used one fixed six-factor
router trained with perturbation consistency, full 64-way composite Renyi-2 balance,
and confidence pressure. Its mean capacity curve was monotone, but seed 9 had a small
1 -> 4 regression:

`2.29863 -> 2.29886`.

That negative result is preserved in `docs/g3b_learned_causal_routing.md` and is not
replaced or hidden.

The failure suggested a specific architectural issue: the full six-factor address can
be globally balanced and stable while the prefixes used by smaller capacities are not
the most reliable nested partitions.

## Predeclared refinement

Before rerunning page-training seeds 7/8/9, the new protocol was fixed as follows:

1. keep the same frozen two-block G3a backbone and internal insertion point;
2. keep one maximum-width learned `48 -> 64 -> 6` router;
3. train the router only from causal block-1 hidden states;
4. use two noisy views of each hidden state for perturbation consistency;
5. apply equal composite-address Renyi-2 deficit pressure to the raw 2-, 4-, and
   6-factor prefixes during router training;
6. retain the same confidence/discreteness pressure;
7. freeze the router;
8. collect a separate deterministic **training-only** hidden-state bank;
9. measure each factor's hard-bit agreement between clean and perturbed hidden states;
10. sort the six factors once by descending perturbation stability, with factor index
    as the deterministic tie break;
11. use the leading 0/2/4/6 factors in that fixed ordering for the 1/4/16/64-page sweep;
12. train page tables independently with the unchanged G3a page-training schedule.

No next-token labels, candidate page losses, semantic page addresses, validation losses,
or future-token information are used to train or order the router.

The learned factor order for the replicated run was:

`[4, 1, 5, 0, 3, 2]`.

Training-only per-factor clean-vs-perturbed stability before ordering was approximately:

| factor | stability |
|---:|---:|
| 0 | 0.98802 |
| 1 | 0.98898 |
| 2 | 0.97342 |
| 3 | 0.97563 |
| 4 | 0.99627 |
| 5 | 0.98870 |

## Fixed resource envelope

The capacity sweep keeps fixed:

- frozen Transformer backbone;
- internal placement: `block 1 -> ParamProbe -> block 2`;
- router architecture: `48 -> 64 -> 6`;
- router learned parameters: `3,526`;
- router matrix MACs/token: `3,456`;
- factor-order metadata: one fixed permutation of six factor indices;
- page operator: `48 -> 10 -> 48` tanh residual MLP;
- page learned parameters: `1,018` FP32 values;
- learned page payload: `4,072` bytes;
- physical external block: `4,096` bytes;
- active page matrix MACs/token: `960`;
- `q_inference = 1`;
- logical external parameter traffic: exactly `4,096 bytes/token`;
- page-training steps, batch size, optimizer, and learning rate.

Only the number of external pages changes across `N = 1, 4, 16, 64`.

## Three-seed result

Frozen-backbone validation CE:

`2.30279203`.

Per-seed validation CE:

| pages | seed 7 | seed 8 | seed 9 |
|---:|---:|---:|---:|
| 1 | 2.29942780 | 2.30025980 | 2.29863489 |
| 4 | 2.29789672 | 2.29822954 | 2.29794645 |
| 16 | 2.29456933 | 2.29397528 | 2.29414484 |
| 64 | **2.29139757** | **2.29163652** | **2.29190350** |

All three individual seeds are strictly monotone:

`1 page > 4 pages > 16 pages > 64 pages` in validation cross-entropy.

Mean +/- sample standard deviation:

| pages | validation CE mean +/- sample std |
|---:|---:|
| 1 | `2.29944083 +/- 0.00081253` |
| 4 | `2.29802424 +/- 0.00017953` |
| 16 | `2.29422982 +/- 0.00030601` |
| 64 | **`2.29164586 +/- 0.00025309`** |

The 64-page mean is about `0.00356` CE lower than the G3a fixed-hash 64-page mean
(`2.29521`) while the active page and external probe budget are unchanged.

## Routing diagnostics

Validation hard-route diagnostics for the fixed ordered router:

| pages | normalized utilization entropy | dead-page fraction | perturbation stability |
|---:|---:|---:|---:|
| 1 | 1.00000 | 0.00000 | 1.00000 |
| 4 | 0.99512 | 0.00000 | 0.98454 |
| 16 | 0.98118 | 0.00000 | 0.96061 |
| 64 | 0.97234 | 0.00000 | 0.90765 |

No validation page is dead at any multi-page capacity.

## Interpretation

This refinement passes the intended G3b architectural question:

> A learned causal factorized router can preserve monotone language-model improvement
> as total external page capacity grows while one 4 KiB page is selected per token,
> with fixed router architecture/compute and a fixed active page operator.

The result does **not** establish that prefix balance or reliability ordering is a novel
routing method. Both should be presented as training/storage compatibility techniques
inside the ParamProbe architecture, not as standalone novelty claims.

The negative first reconstruction remains informative: balancing only the full address
was not sufficient to make every low-capacity prefix robust across seeds. The new result
suggests that nested-capacity experiments benefit when the smaller prefix partitions are
explicitly balanced and the earliest address factors are chosen for perturbation
reliability.

## Research-discipline caveat

This protocol was designed after observing the seed-9 failure of the first shared-router
reconstruction. It is therefore a **new follow-up protocol**, not a post-hoc reinterpretation
of that failed run. The exact refinement was fixed before its 7/8/9 page sweep, and all
three resulting seeds are reported.

Do not keep tuning on this local corpus. The next meaningful falsification step is to
carry the fixed protocol to a named small language benchmark and test whether the
capacity curve survives there.

See `experiments/g3b_prefix_reliable_lm.py`.
