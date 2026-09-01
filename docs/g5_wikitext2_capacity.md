# G5 — WikiText-2 raw frozen capacity replication

## Status

**MEAN CAPACITY CURVES REPLICATE; STRICT PER-SEED MONOTONICITY GATES DO NOT PASS.**

G5 is the first second-corpus test of the frozen paired Tiny Shakespeare protocol. No model, router, page, optimizer, seed, or capacity-sweep hyperparameter was retuned after observing the Tiny Shakespeare results.

The experiment changes only the corpus and split source: it uses the official WikiText-2 raw training and validation files.

## Dataset provenance

The historical Salesforce/MetaMind S3 URL returned a 487-byte error document on the first workflow attempt, before any model training. That attempt is infrastructure-only and contains no scientific result.

The successful run downloads a byte-identical mirror of the original `wikitext-2-raw-v1.zip` archive and hard-checks the original published archive identity before extraction:

- archive bytes: `4,721,645`;
- archive SHA-256: `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.

Extracted split identities:

- `wiki.train.raw`: 10,940,747 bytes, SHA-256 `6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57`;
- `wiki.valid.raw`: 1,146,846 bytes, SHA-256 `4cd0f6876d07a413aa911261ff6d363c72d757d47f0fdd6015702014c89cb9c7`;
- `wiki.test.raw`: 1,290,590 bytes, SHA-256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`.

The experiment uses `wiki.train.raw` for training and `wiki.valid.raw` for validation. The test split is not used.

## Frozen protocol

The architecture and optimization protocol are unchanged from paired Tiny Shakespeare G3f:

- byte-level 256-symbol LM;
- two-block Transformer, `d_model=48`, 4 heads, context 64;
- fresh deterministic backbone, 300 pretraining steps;
- insertion: `embedding -> block 1 -> conditional page -> block 2 -> LM head`;
- page operator: `48 -> 10 -> 48`;
- 1,018 FP32 page parameters;
- 4,072 learned page bytes padded to a 4,096-byte physical block;
- 960 active page matrix MACs/token;
- `q=1`;
- exactly 4,096 logical external parameter bytes/token;
- maximum address width fixed at 8 factors;
- capacities: `N = 1, 4, 16, 64, 256`;
- paired page-prefix initialization and paired training minibatches;
- page seeds 7, 8, 9.

The fixed router executes the full 48x8 projection at every `N`: 384 matrix MACs/token.

The learned router executes the full `48 -> 64 -> 8` router at every `N`: 3,584 matrix MACs/token. Its WikiText-2 training-only perturbation reliability order is

`[7, 2, 6, 3, 5, 4, 0, 1]`.

Frozen-backbone validation CE is `2.48954759`.

## Fixed-hash result

Per-seed validation CE:

| pages | seed 7 | seed 8 | seed 9 | mean ± sample std |
|---:|---:|---:|---:|---:|
| 1 | 2.48249805 | 2.48238274 | 2.48224494 | 2.48237525 ± 0.00012672 |
| 4 | 2.48182876 | 2.48254654 | 2.48187833 | 2.48208454 ± 0.00040087 |
| 16 | 2.48148496 | 2.48109233 | 2.48153384 | 2.48137037 ± 0.00024203 |
| 64 | 2.48109742 | 2.48072171 | 2.48089317 | 2.48090410 ± 0.00018809 |
| 256 | **2.48005520** | **2.48018034** | **2.48015134** | **2.48012896 ± 0.00006551** |

The mean curve is strictly monotone through 256 pages. Every seed has lower CE at 256 pages than at one page:

- seed 7: `-0.00244285` CE;
- seed 8: `-0.00220240` CE;
- seed 9: `-0.00209360` CE.

However, seed 8 has a local `1 -> 4` regression of `+0.00016380` CE. Therefore the predeclared strict criterion—every adjacent capacity step improves in every seed—**does not pass**.

The three-seed mean improvement from 1 to 256 pages is `-0.00224628` CE, corresponding to about a 0.224% relative perplexity reduction.

## Learned-router result

Per-seed validation CE:

| pages | seed 7 | seed 8 | seed 9 | mean ± sample std |
|---:|---:|---:|---:|---:|
| 1 | 2.48249805 | 2.48238274 | 2.48224494 | 2.48237525 ± 0.00012672 |
| 4 | 2.48156795 | 2.48098663 | 2.48159119 | 2.48138192 ± 0.00034253 |
| 16 | 2.47978494 | 2.47974471 | 2.48033535 | 2.47995500 ± 0.00033001 |
| 64 | **2.47887824** | 2.47897844 | 2.47927350 | 2.47904339 ± 0.00020548 |
| 256 | 2.47890259 | **2.47892088** | **2.47891240** | **2.47891196 ± 0.00000916** |

The mean curve is strictly monotone through 256 pages, and every seed has substantially lower CE at 256 pages than at one page:

- seed 7: `-0.00359546` CE;
- seed 8: `-0.00346186` CE;
- seed 9: `-0.00333254` CE.

However, seed 7 has a local `64 -> 256` regression of only `+0.00002435` CE. Therefore the same predeclared strict per-seed monotonicity gate **does not pass**.

The three-seed mean improvement from 1 to 256 pages is `-0.00346329` CE, corresponding to about a 0.346% relative perplexity reduction.

## Cross-corpus interpretation

G5 strengthens the broad capacity-effect evidence while weakening any claim that every individual training run must improve at every capacity increment.

Across both canonical Tiny Shakespeare and WikiText-2 raw:

- the mean fixed-router capacity curve decreases from 1 to 256 pages;
- the mean learned-router curve decreases strongly through 64 pages;
- every WikiText-2 seed improves from 1 to 256 pages for both routing methods;
- the same exact 4 KiB/token external traffic and active page operator are retained.

But the strict adjacent-step gate behaves differently across corpora:

- Tiny Shakespeare paired fixed hash passes all three seeds through 256;
- WikiText-2 fixed hash has one small seed-8 `1 -> 4` reversal;
- Tiny Shakespeare learned routing has two small `64 -> 256` reversals;
- WikiText-2 learned routing has one tiny seed-7 `64 -> 256` reversal.

The correct conclusion is therefore not that a per-seed monotonic law has been established. The supported conclusion is narrower:

> On two named language corpora, increasing inactive page capacity under fixed one-page traffic produces a monotone improvement in the three-seed mean and a lower 256-page loss than the one-page condition in every evaluated seed. Small adjacent-capacity reversals remain possible, especially at the routing-reliability boundary.

This is cross-corpus evidence for a capacity trend, not proof of a deterministic monotonic scaling law.

## Reproducibility

Successful workflow run: `33329027660`.

Artifact:

- name: `g5-wikitext2-results`;
- artifact id: `9737125662`;
- SHA-256: `20cfac3e12a7413e08da9b8d0ebdcfa0da804f6c3996657720a7f2c1b1670360`.

First failed transport-only workflow: `33328928627`; it failed archive verification before model training because the historical S3 endpoint returned 487 bytes.

## Next statistical step

The original three-seed strict gate remains frozen and failed. If more seeds are run, they must be labeled a new **estimation/robustness study**, not used to retroactively redefine this gate. A useful follow-up would predeclare additional independent page-training seeds and estimate paired capacity-step effect distributions and confidence intervals on both named corpora.
