# G3b — learned causal routing replication

## Status

**NOT PASSED.**

The mean validation-loss curve is monotone, and the 16/64-page gains are strong in all three page-training seeds, but the predeclared gate requires monotonicity in every seed. Seed 9 has a small regression from one page to four pages.

This negative result is retained rather than tuned away.

## Repository-state note

The research handoff described an unfinished G3b run, including a strong seed-7 curve, but the corresponding G3b experiment code was never committed to `main`. The repository itself stopped at `experiments/g3a_internal_fixed_hash_lm.py`.

Because the exact uncommitted router hyperparameters and RNG protocol cannot be recovered from Git history, this file records a clean reconstruction that obeys the handoff's architectural constraints more strictly: one learned router is trained once, frozen, and shared across the complete 1/4/16/64 page sweep.

The historical handoff seed-7 values should therefore be preserved as an earlier uncommitted observation, not silently substituted for the replicated numbers below.

## Question

Can the successful G3a internal ParamProbe placement keep improving as external page capacity increases when the fixed hash is replaced by a learned **causal** router that receives no page-address labels and no realized next-token oracle target?

## Architecture

The frozen language model is unchanged from G3a:

`embedding -> block 1 -> ParamProbe -> block 2 -> LM head`

The ParamProbe page operator is also unchanged:

`48 -> 10 -> 48`

Each page has:

- 1,018 FP32 learned parameters;
- 4,072 learned payload bytes;
- one 4,096-byte physical parameter block;
- 960 active page matrix MACs/token;
- exactly one selected page/token (`q=1`).

Therefore logical external parameter traffic is exactly:

`4096 bytes/token`

for 1, 4, 16, and 64 pages.

## Learned causal hash

The reconstructed router is a fixed-width MLP:

`48 -> 64 -> 6`

with GELU activation.

It has 3,526 learned resident parameters and 3,456 matrix MACs/token. The full six-factor router is always executed, including in the smaller-capacity conditions. The 1/4/16/64 conditions use prefixes of the same six hard factors.

The router is trained **once** and then frozen for every capacity point and every page-training seed.

Router training uses only causal block-1 hidden states. It never consumes the next-token target.

For hidden state `h`, two perturbed views are formed:

`h_a = h + eps_a`

`h_b = h + eps_b`.

Let

`p_a = sigmoid(router(h_a))`

and

`p_b = sigmoid(router(h_b))`.

The router objective combines three terms:

1. perturbation consistency: `4 * mean((p_a - p_b)^2)`;
2. full composite-address Rényi-2 deficit using the existing factorized collision estimator;
3. discreteness/confidence pressure: `mean(4 p (1-p))`.

The default weights are:

- consistency: `1.0`;
- composite balance: `0.2`;
- confidence: `0.1`.

The router seed is fixed at 17. Page seeds are 7, 8, and 9.

## Validation diagnostics

The deterministic fallback corpus is the same approximately 1.2 MiB local Python/PyTorch source corpus used by G3a. This is still an architectural diagnostic, not a publication benchmark.

Frozen-backbone validation CE:

`2.30279210`

Router diagnostics on validation hidden states:

| pages | normalized utilization entropy | dead-page fraction | full-address perturbation stability |
|---:|---:|---:|---:|
| 1 | 1.00000 | 0.00000 | 1.00000 |
| 4 | 0.99548 | 0.00000 | 0.97220 |
| 16 | 0.98067 | 0.00000 | 0.92327 |
| 64 | 0.96304 | 0.00000 | 0.90348 |

The stability diagnostic compares the clean hard address with the address after the same feature-scaled perturbation magnitude used in router training. It is evaluated in a separate RNG stream so it cannot change the validation batches used for CE.

## Three-seed page sweep

| pages | seed 7 | seed 8 | seed 9 | mean ± sample std |
|---:|---:|---:|---:|---:|
| 1 | 2.29943 | 2.30026 | **2.29863** | `2.29944 ± 0.00081` |
| 4 | 2.29875 | 2.29898 | 2.29886 | `2.29886 ± 0.00011` |
| 16 | 2.29562 | 2.29495 | 2.29578 | `2.29545 ± 0.00044` |
| 64 | **2.29218** | **2.29261** | **2.29269** | **`2.29249 ± 0.00027`** |

Per-seed monotonicity:

- seed 7: PASS;
- seed 8: PASS;
- seed 9: **FAIL** because `2.29863 -> 2.29886` from one to four pages.

The failure margin is small (`~2.23e-4` CE), but the gate is intentionally strict. G3b is therefore **not passed**.

## Interpretation

Several parts of the learned-router hypothesis survive this replication:

- the router can be trained without next-token labels;
- no validation page is dead;
- full-address utilization remains high;
- the 16- and 64-page conditions beat the one-page control in every seed;
- the mean curve is strongly monotone;
- the 64-page learned-router condition is substantially better than the G3a fixed-hash 64-page mean while preserving the same 4 KiB probe budget.

But the intended claim is stronger than a monotone mean curve. Since seed 9 fails at 1 -> 4 pages, the current learned causal hash is not yet robust enough to satisfy the gate.

## Historical seed-7 handoff observation

The prior research handoff recorded an uncommitted seed-7 result of approximately:

| pages | validation CE |
|---:|---:|
| 1 | 2.29943 |
| 4 | 2.29787 |
| 16 | 2.29436 |
| 64 | 2.29212 |

The reconstructed shared-router protocol reproduces the one-page control exactly and the 64-page value to roughly `6e-5`, but it does not reproduce the intermediate 4/16-page values. Because the source experiment was not committed, those historical numbers are not used to declare the present gate passed.

## Next research move

Do not jump to a large LM and do not erase the seed-9 failure.

The next useful G3b work is to understand the low-capacity 1/4-page variance under a predeclared router modification that remains causal and keeps the same fixed inference resources. Candidate investigations include prefix-aware composite balancing or a reliability-aware factor ordering, but any such change should be treated as a new protocol and rerun from scratch across all seeds.

See `experiments/g3b_learned_causal_hash_lm.py` for the exact reconstructed experiment.
