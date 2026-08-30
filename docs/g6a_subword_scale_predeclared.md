# G6a: predeclared larger subword-LM capacity gate

Status before execution: **PREDECLARED / UNRUN**

## Purpose

G3/G5 establish the capacity trend in a very small byte-level Transformer. G6a changes model/tokenization scale while keeping the ParamProbe question clean.

G6a is deliberately a three-point fixed-router smoke gate. It tests whether the capacity effect survives a materially larger, subword-tokenized causal LM before spending compute on a full five-point sweep or learned-routing study.

No G6a hyperparameter below may be changed after observing G6a capacity results. A changed setting becomes a separately named follow-up and does not rewrite this gate.

## Dataset

Use the same verified WikiText-2 raw official train/validation split as G5/G5b.

Archive identity remains:

- bytes: `4,721,645`;
- SHA-256: `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.

Tokenizer training uses **only `wiki.train.raw`**. Validation text never participates in tokenizer fitting or hash calibration.

## Tokenizer

Use Hugging Face `tokenizers==0.23.1` with a train-only byte-level BPE:

- vocabulary target: `1024`;
- BPE minimum frequency: `2`;
- byte-level pre-tokenizer and decoder;
- byte alphabet included in the initial alphabet;
- one special token: `<unk>`;
- no validation text during tokenizer training.

Save the tokenizer JSON and record its byte size and SHA-256 in the workflow artifact.

This is a corpus-trained byte-level BPE, not GPT-2's pretrained vocabulary. The purpose is to move beyond a 256-symbol byte LM without introducing a large pretrained tokenizer/model dependency.

## Backbone

Causal Transformer:

- vocabulary: tokenizer-produced vocabulary, expected `1024`;
- `d_model = 96`;
- heads: `4`;
- layers: `3`;
- feed-forward width: `384`;
- context length: `128` tokens;
- dropout: `0`;
- pre-norm Transformer encoder layers used causally;
- tied input/output token embeddings;
- ParamProbe insertion after transformer block 2 and before block 3;
- backbone seed: `7`.

Backbone training:

- AdamW;
- learning rate: `1e-3`;
- weight decay: `0.01`;
- gradient clipping: `1.0`;
- steps: `500`;
- batch size: `8` sequences;
- all page sweeps reuse the identical frozen checkpoint.

Record total backbone parameter count and validation CE before adding pages.

## ParamProbe page

Use the 16 KiB page size selected by the earlier Tiny Shakespeare granularity study, but treat the G6a result independently.

Page operator:

- residual MLP `96 -> 20 -> 96`;
- tanh nonlinearity and residual scale `0.15`;
- FP32 learned parameters/page: `3,956`;
- learned payload: `15,824` bytes;
- physical block: `16,384` bytes;
- padding: `560` bytes/page;
- active page matrix MACs/token: `3,840`;
- `q = 1`;
- logical external parameter traffic: exactly `16,384` bytes/token, independent of `N`.

## Router

G6a uses only the deterministic balanced fixed factorized hash, to isolate capacity from learned-router optimization/reliability.

- maximum address width allocated once: `8` bits;
- projection seed: `999`;
- one orthonormal `96 x 8` projection reused for all capacities;
- per-factor thresholds are train-only medians;
- calibration batches: `40`;
- calibration batch size: `8`;
- fixed router matrix MACs/token: `768`;
- smaller capacities use prefixes of the same eight-factor address.

## Capacity sweep

Predeclared G6a points:

- `N=1` (`used_bits=0`);
- `N=16` (`used_bits=4`);
- `N=256` (`used_bits=8`).

External learned payload capacity therefore grows from 15,824 bytes to 4,050,944 bytes while selected traffic and active page compute remain fixed.

## Page training

Independent page-training seeds: `7, 8, 9`.

For each seed/capacity:

- page-prefix initialization is paired across capacities;
- dedicated initialization seed base: `40000 + seed`;
- minibatch RNG is reset after page initialization;
- dedicated page-batch seed base: `50000 + seed`;
- the same minibatch sequence is therefore used for `N=1,16,256` for a given seed;
- backbone remains frozen;
- AdamW learning rate: `4e-3`;
- weight decay: `1e-4`;
- page-training steps: `120`;
- batch size: `8` sequences.

## Evaluation

- validation split only;
- evaluation RNG seed: `1234`;
- evaluation steps: `40`;
- evaluation batch size: `8`;
- identical evaluation batches for every capacity/seed;
- report CE mean over evaluation batches;
- also report normalized route-utilization entropy and dead-page fraction.

## G6a pass/fail criterion

The scientific gate passes only if all of the following hold without rerunning/tuning:

1. resource assertions hold exactly: `q=1`, one 16 KiB selected block/token, fixed active page MACs, fixed maximum-width router resources;
2. every page seed is strictly capacity-monotone over the predeclared three points:
   `CE(N=1) > CE(N=16) > CE(N=256)`;
3. no validation information is used in tokenizer fitting, routing thresholds, page initialization, or page training.

Mean monotonicity alone is not sufficient for this strict smoke gate, though all numbers will be retained if the gate fails.

## After G6a

If G6a passes, G6b will be declared separately before execution and will expand to `N=1,4,16,64,256` and add a learned factorized router under a fixed maximum width.

If G6a fails, the result is frozen. Any changed training duration, model size, tokenizer, page size, insertion point, or optimizer becomes a separately named experiment rather than a reinterpretation of G6a.
