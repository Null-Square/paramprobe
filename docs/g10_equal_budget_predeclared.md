# G10: bounded equal-payload-training and progressive-expansion test

Date: 2026-09-12. This protocol is committed before new G10 model training. It does not amend G9b. Base research commit: debd4a5005c9560b9c7650ab524fb6563bf74dd3.

## Question and fixed design

Does exact function-preserving expansion of 16 trained pages into 256 pages improve held-out quality over both independently trained 16-page and 256-page models at the same payload-training step/token budget?

Use the hash-pinned G6a 446,304-parameter backbone, tokenizer and WikiText-2 raw splits. All six fresh seeds are 47,48,49,50,51,52. Each seed uses the existing G6a hidden-bank implementation, 1,920 batches of 8 sequences of length 128, bank seed 50,000 + seed. Initialization uses G6c init_pages (seed 40,000 + seed). Fixed routing uses the existing train-only median-calibrated eight-bit projection (40 batches, batch 8). All used_bits settings compute the same eight projection dimensions; lower bits are discarded only after projection.

Methods:
1. fixed16: original fixed router, 16 pages, 1,920 AdamW steps.
2. fixed256: original fixed router, 256 pages, 1,920 AdamW steps.
3. clone256: train exactly as fixed16 through step 120, then copy each parent into 16 independently stored child pages; retain prefix-parent mapping child_id // 16. Copy AdamW first/second moments and step counters as well as weights. Complete 1,800 additional steps. No new routing or residual gate; no noise, hyperparameter search, or split-time sweep.
4. balanced_pk256: unchanged G7b balanced product-key router and payload mechanism with this seed and the same 1,920 payload batches. Its 600-step router pretraining is EXTRA and will be reported, not mislabeled iso-total-training-compute.

Learning rate 0.004, weight decay 0.0001; other AdamW defaults as existing G6c. Page shape 96 -> 20 -> 96, tanh activations, scale 0.15, 15,824 payload bytes inside a 16,384-byte block. One selected page per token. In-memory training is NOT a physical SSD latency experiment. Equal payload steps/token exposure and page MACs are NOT a claim of equal total training FLOPs, memory or wall time.

Checkpoints: 120, 480, 960, 1,920 steps; diagnostic validation bank is unchanged (40 x 8 x 128; seed 1234). Results at every recorded checkpoint will be retained, not selected retrospectively. The fixed16 prefix can be reused to create clone256, but 120 shared steps are charged to each logical method. Initial clone equality must be checked numerically on a training minibatch before further optimization.

Primary final endpoint: complete WikiText-2 TEST split, each next-token target scored once with context reset every 128 tokens, including the final partial segment. The protocol uses the test split once for these pre-fixed comparisons; no tuning or follow-up selection on these test results. Validation and test scores with different context coverage must not be compared as identical metrics. Save final model states and per-segment losses. No retraining of the backbone or tokenizer.

## Decision and stopping rules

A survivor requires clone256 to reduce mean test CE by >= 0.005 nats/token versus EACH fixed-capacity baseline, with a win in all six paired seeds. The 0.005 threshold is a project decision threshold, not an established universal practical significance cutoff. Six same-direction seed wins have a one-sided exact sign-test p=1/64 under independent equally likely signs; two predeclared directional comparisons have Bonferroni bound 2/64. This measures seed-level directional reproducibility on ONE frozen backbone/corpus, not broad model/dataset generalization.

Additionally report comparison with balanced_pk256 without hiding its extra router-training cost. A candidate not better than this strong comparator is not evidence of a new quality/resource frontier. If the practical threshold fails, STOP promoting this particular clone-expansion method; do not tune the split point in this pass or add an unplanned sweep. A positive result is only a candidate mechanism result, not proof of priority or submission readiness. A negative result closes this candidate, not every possible external-memory research direction.

Interrupted executions must remain explicitly partial. No favorable seeds may be dropped. Record source hashes, asset hashes, environment, timings and any deviations. Existing G9b classification must be reported independently under its unchanged original gate.
