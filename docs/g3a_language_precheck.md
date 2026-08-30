# G3a — internal language-model capacity precheck

## Question

Does the ParamProbe capacity effect survive a language-model setting when the external page is a genuine nonlinear operator, inference reads exactly one 4 KiB page per token, and only inactive external capacity grows?

G3a is deliberately a **precheck**, not a benchmark result. The execution container could not download the full Tiny Shakespeare file, so the reported run uses a deterministic 1.2 MiB concatenation of local Python/PyTorch source files. The experiment script accepts `--text-path` for a standard corpus and publication-grade runs must use a named dataset.

## First attempt: ParamProbe immediately before the LM head

The first frozen-backbone experiment placed the page residual immediately before the language-model head. It did **not** show monotone scaling. A representative validation cross-entropy sweep was approximately:

| pages | validation CE |
|---:|---:|
| frozen backbone | 2.393 |
| 1 | 2.372 |
| 4 | **2.357** |
| 16 | 2.372 |
| 64 | 2.375 |

This negative result was retained rather than tuned away.

### Route-gap diagnosis

For the pre-head architecture, evaluating the same fixed four training-time candidate routes with the realized next-token label produced a very large oracle route gap:

- 4 pages: about `0.89` CE;
- 16 pages: about `1.03` CE;
- 64 pages: about `1.23` CE.

The page residual functions themselves were diverse, so simple page redundancy was not the initial explanation.

However, a separate predictability probe showed that this candidate-wise oracle is partly impossible for a causal router: a 128-hidden-unit MLP trained on the frozen hidden state could predict the realized best page for the 4-page condition only about `48.6%` of the time (majority baseline about `38.0%`), while the learned router already achieved about `47.2%`.

The realized best page depends on the true next token, which is privileged future information. Therefore the full counterfactual oracle gap is **not** a valid measure of attainable routing quality in language modeling.

A training-only expected-utility critic removed that privileged-information target, but then the pages converged to nearly the same function (mean pairwise residual cosine around `0.93`). This exposed a second failure mode: expected-utility routing can eliminate route regret by destroying specialization.

## Stable-router diagnostic

To isolate whether the failure came from learned routing or from the insertion point itself, the next test used a fixed balanced context hash:

1. freeze the pretrained Transformer;
2. project the frozen hidden state through one fixed `48 x 6` random orthogonal matrix;
3. calibrate one median threshold per factor on training hidden states;
4. allocate all six factors once;
5. use prefixes of the same six-bit address for 1/4/16/64-page conditions;
6. train only the external page MLPs.

This removes next-token leakage and keeps router metadata and routing compute literally fixed across the sweep.

The same stable hash still failed to produce monotone scaling when ParamProbe was placed immediately before the head. That motivated moving the layer **inside** the Transformer.

## Internal insertion

The successful precheck uses a frozen two-block byte-level Transformer:

`embedding -> block 1 -> ParamProbe -> block 2 -> LM head`

The downstream Transformer block can interpret the selected external residual before logits are produced.

### Hard resource envelope

Every capacity condition uses:

- `q_inference = 1`;
- `B = 4096 bytes`;
- fixed maximum six-factor hash router;
- router matrix: `48 x 6`, plus six thresholds;
- fixed routing MACs: `48 x 6 = 288` matrix MACs/token;
- page operator: `48 -> 10 -> 48` tanh residual MLP;
- page learned parameters: `10*48 + 10 + 48*10 + 48 = 1018` FP32 values;
- learned page payload: `1018 * 4 = 4072 bytes`;
- each page is padded to exactly one 4096-byte external parameter block;
- active page matrix MACs: `48*10 + 10*48 = 960`/token;
- identical frozen Transformer backbone and LM head;
- identical page training steps and optimizer settings.

Only the number of **inactive external pages** changes.

An earlier internal run accidentally used a 6.4 KiB page and was discarded from the strict probe-budget result. The numbers below are from the corrected 4,072-byte learned payload.

## Replicated result

The frozen-backbone validation CE is:

`2.3027921`.

Three page-training seeds give:

| external pages | validation CE mean ± sample std |
|---:|---:|
| 1 | `2.29944 ± 0.00081` |
| 4 | `2.29751 ± 0.00037` |
| 16 | `2.29629 ± 0.00014` |
| 64 | **`2.29521 ± 0.00028`** |

The ordering is monotone in every individual seed.

Per-seed values:

| pages | seed 7 | seed 8 | seed 9 |
|---:|---:|---:|---:|
| 1 | 2.29943 | 2.30026 | 2.29863 |
| 4 | 2.29709 | 2.29776 | 2.29770 |
| 16 | 2.29645 | 2.29624 | 2.29618 |
| 64 | 2.29496 | 2.29517 | 2.29551 |

No validation page is dead in these sweeps. The normalized hard-route utilization entropy for the fixed hash decreases from about `0.962` at four pages to about `0.820` at 64 pages, so the result does not depend on perfectly uniform routing.

## Interpretation

G3a provides the first language-model evidence for the core ParamProbe capacity hypothesis:

> With a fixed frozen backbone, fixed resident routing metadata, fixed active page operator, fixed active compute, and exactly one 4 KiB external parameter block selected per token, increasing only the number of available external learned pages improves held-out next-token loss.

The effect is small. It should not be advertised as a competitive language-model improvement. Its value is that the controlled capacity curve survives the transition from synthetic regression to causal language modeling.

The insertion-point negative control is equally important: page capacity was not consistently useful when the external residual was attached directly before the output head. Downstream computation appears to matter.

## What G3a does *not* establish

G3a does not establish:

- task-trained factorized routing in language modeling;
- superiority over MoE, memory layers, adapters, or PEER;
- scaling on a standard named LM benchmark;
- physical SSD/flash latency for token-by-token language inference;
- that the improvement continues beyond 64 pages;
- a language-model scaling law.

The fixed hash is a diagnostic router, not the intended final architecture.

## Next gate

G3b should keep the successful internal insertion and replace the fixed hash with a trainable factorized router **without using realized next-token oracle labels as routing targets**.

The key methodological constraint is that routing supervision must be causal/predictable from the hidden state. Candidate losses may train a value/utility estimator, but the router target cannot directly depend on which page happened to assign the largest probability to the realized future token.

A G3b pass requires the 1/4/16/64 capacity curve to remain monotone across multiple seeds while preserving:

- one 4096-byte page probe/token;
- fixed maximum-width router architecture and MACs;
- the same 1018-parameter active page MLP;
- fixed frozen backbone for the first learned-router comparison.

See `experiments/g3a_internal_fixed_hash_lm.py` for the reference implementation.
