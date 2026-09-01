# G9a predeclaration: local-KV hard-usage diagnostic

Status before execution: **PREDECLARED / UNRUN**

## Purpose

G9's resource-matched local KV block lost to the nonlinear operator in all three fresh seeds. Its reported normalized per-query local-softmax entropy was approximately zero, but that statistic measures retrieval sharpness, not whether the same local slot is reused for every query.

G9a is therefore **diagnostic-only**. It must reproduce G9 training unchanged and add hard local-slot usage instrumentation after training. No optimizer, initialization, model, router, payload shape, seed, data, resource budget, or interpretation criterion from G9 is changed.

## Frozen G9 protocol

Use exactly the G9 assets and settings:

- exact archived G6a tokenizer/backbone hashes;
- official checksum-pinned WikiText-2 raw data;
- seeds 38, 39, 40;
- N=256 blocks;
- q=1 physical 16,384-byte block/token;
- shared balanced product-key d_key=6 router trained exactly as G7b and frozen before payload training;
- local KV block = 21 x 96-D keys + 21 x 96-D values;
- operator block = 96 -> 20 -> 96;
- payload training = 1,920 minibatches x 8 sequences x 128 tokens with seed 50000 + experiment seed;
- validation bank = 40 x 8 x 128, seed 1234.

The reproduced G9 CE values may differ only by ordinary floating-point execution noise; G9a is not a new quality comparison and does not replace the frozen G9 result.

## New diagnostic metrics only

After the KV payload for each seed is fully trained and frozen, evaluate the local 21-way key scores on the **same 1,920-batch training bank** and separately on the frozen validation bank.

For each query, record the hard local slot `argmax_j <x, k_j>` within its already-selected global block.

Treat each `(global_block_id, local_slot_id)` pair as one of 256 x 21 = 5,376 physical local KV slots.

Report, separately for training and validation:

1. global hard local-slot utilization entropy, normalized by `log(5376)`;
2. global dead local-slot fraction among the 5,376 `(block,slot)` pairs;
3. number of hard-active local slots;
4. traffic-weighted mean per-block normalized local hard-usage entropy, where a block's entropy is normalized by `log(21)`;
5. median per-block active local-slot fraction;
6. 10th-percentile per-block active local-slot fraction;
7. maximum traffic share assigned to any single local slot within each block, summarized by median and 90th percentile across blocks.

Blocks with zero observations in a given evaluation bank are excluded only from per-block summaries, never from the global dead-slot statistic.

Also continue reporting the original mean per-query softmax entropy so sharpness and usage are explicitly separated.

## Predeclared interpretation

G9a does not change the G9 classification.

For deciding whether one stronger KV anti-collapse follow-up is scientifically warranted, use the much larger **training-bank** hard-usage statistics, because every block has far more observations there than on validation.

Define `local_usage_healthy=true` only if, in every seed:

- normalized global hard local-slot utilization entropy >= 0.85; and
- global dead local-slot fraction <= 0.05.

If `local_usage_healthy=true` in all three seeds, no anti-collapse KV follow-up is allowed on this corpus; the low G9 softmax entropy is treated as confident sparse retrieval rather than evidence of collapse.

If either criterion fails in any seed, exactly one G9b training-only anti-collapse KV protocol may be predeclared before execution. G9b must keep the exact same inference payload shape, q=1, B=16 KiB, router, learned bytes, and active dot/weighted-sum MAC envelope. No second KV tuning follow-up is allowed after G9b.

These thresholds mirror the anti-collapse discipline used in G7b and are fixed before G9a execution.
