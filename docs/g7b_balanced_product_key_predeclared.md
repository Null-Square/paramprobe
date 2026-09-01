# G7b: predeclared balanced q=1 product-key challenge

Status before implementation/execution: **PREDECLARED / UNRUN**

## Motivation

G7's predeclared resource-matched product-key adversary (`d_key=6`) did not beat the fixed factorized router, but it also exhibited severe utilization collapse: roughly 84–88% dead pages on validation. That makes G7 a valid negative baseline result but not a decisive architectural comparison against a well-utilized q=1 product-key router.

G7b is the **single planned anti-collapse follow-up**. It changes product-key **training only**. Inference resources, page shape, product-key dimension, top-1 address rule, and evaluation protocol remain unchanged.

G7 is frozen and cannot be reclassified by G7b.

## Frozen backbone/data/operator

Use the exact G6a assets:

- tokenizer SHA-256 `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`;
- backbone SHA-256 `a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049`;
- verified WikiText-2 raw official train/validation split, archive SHA-256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`;
- 1,024-token train-only BPE;
- frozen `d_model=96`, 3-layer, context-128 causal LM, insertion after block 2.

External operator remains exactly:

- `N=256` pages;
- `96 -> 20 -> 96`, tanh, residual scale 0.15;
- 3,956 FP32 parameters/page = 15,824 learned bytes;
- one 16,384-byte physical page;
- active page matrix compute 3,840 MACs/token.

## Inference contract

Both methods satisfy:

- `q=1`;
- exactly one selected 16 KiB page/token;
- exactly 16,384 logical external parameter bytes/token;
- only the selected page operator executes;
- no training-time regularizer or probability table is retained as capacity-dependent inference state.

### Fixed factorized reference

Unchanged G6/G7 fixed 8-bit router:

- 3,104 bytes fixed routing metadata;
- 768 matrix MACs/token;
- no trainable router.

### G7b product-key router

Exactly the G7 resource-matched `d_key=6` router:

- learned query `96 -> 6`;
- two 16x3 product-key codebooks;
- non-affine `BatchNorm1d(6)`;
- top-1 Cartesian address from independent subkey argmaxes;
- inference gate `sigmoid(best_score_1 + best_score_2)`;
- 678 learned FP32 router parameters = 2,712 bytes;
- 56 bytes BatchNorm running buffers;
- 672 query/key-score MACs/token.

Thus G7b product-key inference remains below the fixed factorized router in both learned/fixed router bytes and matrix/dot MACs at N=256.

## Training-only anti-collapse protocol

The anti-collapse router objective is fixed by **transferring the previously used G3b router-training hyperparameters**, rather than tuning against G7 validation loss:

- router steps: 600;
- router batch size: 1,024 hidden states;
- AdamW lr `2e-3`, weight decay `1e-4`;
- feature-scaled Gaussian perturbation std `0.08`;
- consistency weight `1.0`;
- balance weight `0.2`;
- confidence weight `0.1`.

Training-only hidden-state bank:

- 80 frozen G6 backbone batches x 8 sequences x 128 tokens;
- WikiText-2 **training split only**;
- hidden-bank RNG seed 2468;
- flattened bank contains 81,920 causal insertion-point hidden states.

For fresh seed `s`, initialize the product-key router exactly as in G7 with seed `60000+s`, then set router-training RNG seed `71000+s`.

### Categorical product-key regularizers

For a clean/noisy hidden vector, compute the two 16-way subkey softmax distributions. Their outer product is the differentiable 256-page product distribution `p`.

For two independent perturbations `p_a`, `p_b`, define `p_bar=(p_a+p_b)/2` and:

1. **Consistency**: `256 * mean((p_a - p_b)^2)`.
2. **Composite balance**: `KL(mean_batch(p_bar) || Uniform(256)) = log(256) - H(mean_batch(p_bar))`.
3. **Confidence**: `mean(H(p_bar)) / log(256)`.

Router-only loss:

`L_router = 1.0*consistency + 0.2*balance + 0.1*confidence`.

No LM labels, next-token loss, validation examples, semantic addresses, realized page losses, or counterfactual page targets are used for this 600-step router training.

After 600 steps the product-key router is **frozen**. It is not updated during page training. This makes G7b an apples-to-apples comparison of two label-free routing partitions feeding the same trained external page operators.

## Page training

Fresh page/router seeds: **35, 36, 37**.

For each seed and both router methods:

- page initialization seed `40000+s`, identical across methods;
- page hidden-state/target bank: 1,920 minibatches x 8 x 128, seed `50000+s`, identical across methods;
- AdamW page lr `4e-3`, weight decay `1e-4`;
- pages train for exactly 1,920 steps;
- router remains frozen during page training.

Training-resource accounting reports the extra 600 product-key router-only steps separately. This changes training cost, not inference cost.

## Evaluation

Use the same official WikiText-2 validation protocol as G6/G7:

- frozen 40 x 8 x 128 evaluation hidden-state bank;
- RNG seed 1234;
- report CE, normalized utilization entropy, dead-page fraction, mean inference gate, and clean-vs-feature-noise top-1 route stability at noise std 0.08.

No validation result is used for training or model selection.

## Predeclared utilization gate

G7b is considered a meaningful anti-collapse comparison only if **every seed 35/36/37** has:

- normalized validation utilization entropy `>= 0.85`; and
- dead-page fraction `<= 0.05`.

If this utilization gate fails in any seed, G7b is classified **INCONCLUSIVE / ANTI-COLLAPSE FAILED** regardless of CE.

## Predeclared finite-N interpretation

Only if the utilization gate passes:

1. If balanced product-key has lower CE than fixed factorized routing in **all three seeds and in the mean**, then the finite-N claim that the fixed factorized router occupies a uniquely favorable q=1 quality/resource point is **killed**. The surviving distinction is the formal block-probe framework and asymptotic router scaling (`log N`-style factorized versus product-key `sqrt(N)`).
2. If fixed factorized routing has lower CE in **all three seeds and in the mean**, this is stronger evidence that the current fixed factorized partition is a better finite-N q=1 quality/resource point than a non-collapsed resource-matched product-key partition. It is still not a claim that PEER itself is inferior.
3. Any mixed per-seed outcome is **UNRESOLVED**.

No G7b hyperparameter, seed, utilization threshold, or interpretation may be changed after execution begins.
