# G6e: predeclared paired low-capacity edge robustness study

Status before execution: **PREDECLARED / UNRUN**

## Purpose

G6d is frozen as a failed strict mean-monotonicity gate. Its only mean reversal is the very small `N=4 -> N=16` edge (`+0.00004278` CE), while `N=256` beats both `N=1` and `N=16` in every fresh G6d seed and the `16 -> 64 -> 256` mean improvements are much larger.

G6e is a new estimation study declared after G6d. It does not rerun or redefine G6d. It isolates only the ambiguous low-capacity `N=4` versus `N=16` comparison under the exact same 120-step protocol and estimates its paired effect across many fresh page-training seeds.

## Frozen assets and model

Use the exact archived G6a assets:

- tokenizer SHA-256: `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`;
- backbone SHA-256: `a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049`;
- WikiText-2 raw archive SHA-256: `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.

Frozen configuration is identical to G6a/G6d:

- byte-level BPE vocabulary 1,024;
- causal Transformer `d_model=96`, 3 layers, 4 heads, FF 384, context 128;
- insertion after block 2;
- fixed balanced maximum-width 8-factor hash, projection seed 999;
- `q=1`, physical block 16,384 bytes, logical external parameter traffic 16,384 bytes/token;
- page operator `96 -> 20 -> 96`, 3,956 FP32 parameters / 15,824 learned bytes;
- active page matrix MACs/token 3,840;
- fixed maximum-width router matrix MACs/token 768.

## Fresh seeds and paired protocol

Use 16 fresh page-training seeds **16..31**, selected before execution.

For each seed:

- page initialization seed: `40000 + seed`;
- hidden-state training-bank seed: `50000 + seed`;
- materialize exactly 120 hidden/target minibatches using the frozen G6a hidden-bank implementation;
- train `N=4` and `N=16` on the identical 120-minibatch bank;
- explicitly reinitialize page tables after construction with the same seed, so shared prefix page rows use paired initial values;
- AdamW, lr `4e-3`, weight decay `1e-4`;
- exactly 120 training steps for both capacities.

Thus total routed training assignments are equal: `120 * 8 * 128 = 122,880` for both conditions. Mean assignments/page differ exactly as in G6d: 30,720 for N=4 and 7,680 for N=16.

## Evaluation

Reuse the frozen G6a evaluation bank:

- official WikiText-2 raw validation split;
- seed 1234;
- 40 batches x 8 sequences;
- context 128;
- same evaluation hidden-state bank for every seed and capacity.

Report CE for N=4 and N=16, paired delta `delta = CE16 - CE4`, sign counts, mean delta, sample standard deviation, standard error, and a two-sided 95% Student-t interval using df=15 (`t_0.975 = 2.1314495456`).

## Predeclared interpretation

This is an estimation study, not a replacement pass/fail gate.

- If the entire 95% interval for paired `CE16 - CE4` is below zero, conclude the broader fresh-seed evidence supports N=16 over N=4 under the frozen 120-step protocol.
- If the entire interval is above zero, conclude it supports N=4 over N=16.
- If the interval overlaps zero, classify the low-capacity edge as statistically unresolved at this sample size.

Also report the number of seeds with `CE16 < CE4`.

No outcome from G6e changes the frozen fact that G6d failed its predeclared strict mean-monotonicity criterion.

## Why this is the next diagnostic

G6c already confirms the large-N training-exposure crossover, and G6d already shows strong 16->64->256 improvement under the explicit training rule. Repeating expensive 64/256 conditions would add little information about the only edge that prevented the G6d mean curve from being strictly monotone. G6e therefore spends new computation only on the unresolved comparison.
