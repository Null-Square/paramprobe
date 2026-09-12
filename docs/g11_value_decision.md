# G11: measurable engineering value; stop the standalone architecture-paper track

Date: 12 September 2026.

## Decision

Retain the tested compression/export implementation and evidence. Stop further paper-driven development of the current generic architecture/growth claim. This is an allocation-of-effort decision, not proof that all possible future ParamProbe ideas lack value.

G11 demonstrates useful compression of the existing memory. It does not establish a novel algorithm, contemporary-scale model advantage, real-device speedup, or publication readiness. Ordinary quantization explains the strongest storage result.

## Correction and recovered evidence

The preceding conversation report incorrectly described G9b as unrun and omitted completed GitHub work. This pass recovered and hash-verified the completed runs, source/corpus bundle, original tokenizer/backbone, and three trained KV/router checkpoints.

First G9b closure: run 34688148066, artifact 10296029650. Mean validation CE: operator 5.69858812, balanced KV 5.70076507. All three seeds fail the frozen utilization gate. The recorded classification is `anti_collapse_failed`; `further_kv_tuning_allowed=false`.

Separate functional-audit reproduction: run 34689101948, artifact 10296272701. Its saved checkpoints for seeds 44,45,46 are used by G11. It differs numerically from the first closure; these are not to be merged as identical runs. G11 changes no training setting, weight optimization, G9b gate or classification.

Source/corpus bundle: run 34688056478, artifact 10296296022; source d6280c53e5da3dbd77d46f1a1aab87a674af379d. Corpus archive SHA-256: ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11.

## Fixed deployment-only experiment

Compare original 21-slot FP32 memory, FP16 and elementary blockwise INT8, and training-mass pruning to ten or five slots. These are static learned parameter memories, NOT the autoregressive attention KV cache. Low-precision formats are decoded to FP32; this is not GPTQ or a low-precision arithmetic kernel.

Selection uses archived training probability means only. Each calibration model-state fingerprint was independently verified against the loaded checkpoint. Test labels do not select slots or quantization settings.

The locally fixed protocol hash is 474712e81b90ab66298c542a8eb019e9ad041dd426447028fe400a8542991cce. This is local precommitment, not external preregistration. Its practical criterion is at least 50% fewer requested page bytes with mean test-CE increase at most 0.001 nats in EACH checkpoint seed. No tuning or new training follows the screen.

Official WikiText-2 test data: 504,654 tokens under the original 1,024-entry BPE, 504,576 scored next-token targets in 3,942 consecutive 128-target windows, with context reset at each window. One initial token and 77 trailing targets are unscored. Backbone: 446,304 parameters. All five payload arms use file-backed execution for every scored target; no resident-table quality surrogate.

## Results

Means across archived checkpoint seeds 44,45,46; lower CE is better. The same corpus is reused across seeds.

| Method | Bytes/token | External table | Payload matrix MACs/token | Mean test CE | Delta vs FP32 |
|---|---:|---:|---:|---:|---:|
| Original 21 FP32 | 16384 | 4 MiB | 4032 | 5.703412626 | 0 |
| Full 21 FP16 | 8192 | 2 MiB | 4032 | 5.703412612 | -0.000000014 |
| Full 21 INT8 | 4096 | 1 MiB | 4032 | 5.703416835 | +0.000004209 |
| Keep 10 FP32 | 8192 | 2 MiB | 1920 | 5.703412670 | +0.000000044 |
| Keep 5 FP32 | 4096 | 1 MiB | 960 | 5.703652821 | +0.000240194 |
| Backbone only | 0 | 0 | 0 | 5.711617374 | +0.008204748 |

All compressed arms pass the practical criterion in every checkpoint seed. Removing the memory fails it. The memory therefore contributes useful behavior, but its original byte footprint is unnecessary for nearly the same average loss.

INT8 reduces table and requested page bytes by 75%. Its worst seed mean CE increase is 4.940381e-6 nats; mean relative perplexity increase is approximately 0.00042%. This is a component-level byte saving, NOT 75% less total process RAM or end-to-end latency.

Ten-slot pruning reduces payload matrix MACs by 52.38%; five-slot pruning by 76.19%, with a larger loss change. Matrix counts exclude scalar operations, dequantization, the global router and backbone. No overall inference speedup is claimed.

## Execution and robustness

Total: 7,568,640 explicit pread calls, 62,002,298,880 returned bytes (57.744 GiB), no failed reads. Every query reads one complete page. A 32-byte header, quantization scales and padding are included. No decoded-page cache is used. Ordinary pread permits OS page caching: these are application-read counts, not cold-SSD or physical-device evidence.

Near-identical average loss is not exact equivalence. Worst observed absolute residual-coordinate changes: FP16 0.00110, INT8 0.07091, ten-slot pruning 0.11492, five-slot pruning 0.24887. Largest 128-target-window CE increases: respectively about 7.63e-6, 3.48e-4, 1.23e-4 and 0.00509 nats. These are not per-token worst-case guarantees.

Original FP32 file execution matches the tensor reference on the first checked batch per seed within 1.52e-6 maximum residual-coordinate error. All per-window losses, seed summaries, hashes, read counters, tail diagnostics and descriptive contiguous-block bootstrap sensitivity intervals are preserved in the G11 execution packet. A single corpus and three checkpoints cannot support broad deployment guarantees.

Tests: 42 new tests plus all 10 in the recovered source snapshot pass together (52 passed, zero failed/skipped). A repository-layout integration fixture also passes the same 52 tests; this is overlapping verification, not 104 tests. Later G10 files absent from the recovered snapshot were not part of this full-suite run.

No new training steps, model API calls or remote training workflows were used in this pass.

## Mathematical control

For each tanh coordinate z, softmax(z,-z)[0]-softmax(z,-z)[1]=tanh(z). A separate two-slot group per hidden coordinate, with tied opposite value vectors, exactly reproduces the operator after summing groups and applying the same bias and outer tanh. It needs the same independent stored parameters when sign ties are exploited. Literal softmax scalar costs differ; no speed equivalence is claimed.

This is NOT equivalence to G9's single globally normalized 21-slot memory. It shows that 'operator' versus 'memory' is too broad a category-level novelty claim. Forward/gradient tests and explicit assumptions are supplied.

A pruning audit bound is 2*abs(residual_scale)*dropped_probability_mass*max_value_norm. Computing the dropped mass requires original keys; it is not a free deployed certificate or a downstream loss guarantee. A constructive rank-three probability example also proves that never being argmax does not imply a soft slot is functionally unused.

## Why stop the paper track

The useful gains require no new learning mechanism. The generic growth route is directly covered by Expert Upcycling (arXiv:2604.19835). Feed-forward/key-value memory relationships are established by Geva et al. (arXiv:2012.14913) and normalization studies such as Shen et al. (arXiv:2302.06461). Post-training quantization is mature; GPTQ (arXiv:2210.17323) is one established example, not the algorithm implemented here.

The full-memory benefit over no memory is only about 0.82% relative perplexity in this small setting. There is no larger-model, multiple-dataset, real-device or user-workload advantage. Keep the useful code and honest technical result; do not spend more compute merely to force the current claim into a novel-architecture paper. Reopening requires a concrete unmet workload or a genuinely different hypothesis, not another unbounded capacity sweep.

## Code and reproducibility

This branch adds src/paramprobe/compact_kv.py, src/paramprobe/payload_theory.py and tests/test_g11_value.py. Existing experiment protocols and main remain untouched. The separately delivered G11 execution packet contains the actual driver, frozen protocol, recovered assets/source, all exported tables, complete per-window losses, detailed proofs and a SHA-256 manifest. The branch is not claimed to contain that entire binary execution packet or to have passed remote CI.
