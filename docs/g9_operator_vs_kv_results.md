# G9 result: one-block nonlinear operator vs local KV memory

Status: **COMPLETED — nonlinear operator supported by the predeclared G9 criterion, with one diagnostic caveat**

G9 was predeclared before execution in `docs/g9_operator_vs_kv_block_predeclared.md`.

## Provenance

The scientifically valid execution used the exact archived G6a assets rather than a retrained approximation.

- WikiText-2 raw archive SHA-256: `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`
- tokenizer SHA-256: `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`
- backbone SHA-256: `a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049`
- exact archived G6a artifact SHA-256: `0e901efe5d360b905fbb15c5cdb5d90ffea3ad60d7f05e2712fe43fc37d2ad21`
- fresh seeds: 38, 39, 40

A preceding recovery attempt reproduced the tokenizer and tokenized split lengths exactly and reproduced the 500-step backbone training trace to approximately six decimal places, but the serialized backbone SHA differed. The strict identity gate correctly stopped that attempt before G9. The final execution transported and verified the original frozen checkpoint bytes.

## Shared inference contract

Both methods used:

- `N = 256` external physical blocks;
- `q = 1` physical block/token;
- `B = 16,384` bytes;
- exactly 16,384 logical external bytes/token;
- the same frozen balanced product-key `d_key=6` router within each seed;
- router learned bytes = 2,712 plus 56 bytes BatchNorm buffers;
- router matrix MACs/token = 672;
- the same routed block ID, global gate, frozen backbone, training examples, and validation bank.

Payload A, nonlinear operator:

- `96 -> 20 -> 96` tanh residual MLP;
- 3,956 FP32 parameters/block;
- 15,824 learned bytes/block;
- 3,840 active matrix MACs/token.

Payload B, local KV memory:

- 21 learned 96-D keys + 21 learned 96-D values per block;
- 4,032 FP32 parameters/block;
- 16,128 learned bytes/block;
- 4,032 active dot/weighted-sum MACs/token.

Thus the KV baseline received 304 additional learned bytes/block and 192 additional active MACs/token.

## Exact results

| seed | operator CE | KV-block CE | operator better | global util entropy | global dead fraction |
|---:|---:|---:|:---:|---:|---:|
| 38 | 5.69769732 | 5.70200020 | yes | 0.96225214 | 0.00000000 |
| 39 | 5.69792526 | 5.70183345 | yes | 0.92042560 | 0.00000000 |
| 40 | 5.69873375 | 5.70246364 | yes | 0.94335938 | 0.03515625 |

Means:

- nonlinear operator: `5.69811877 ± 0.00054464` sample std;
- local KV block: `5.70209910 ± 0.00032653` sample std;
- KV minus operator mean CE: `+0.00398032`.

The nonlinear operator beat the local KV block in all three fresh seeds and in mean. The G9 resource gate passed.

Predeclared classification:

`g9_classification=nonlinear_operator_supported`

## Important diagnostic caveat

G9 reported the normalized entropy of the **per-query 21-way local softmax**. It was approximately zero for all three KV runs. This means each query's local retrieval became very sharp; by itself it does **not** prove that all queries select the same local slot or that most local slots are dead.

The experiment did not report hard local-slot usage counts across the 256 x 21 = 5,376 local KV slots. Therefore we must not retroactively label the G9 KV baseline as collapsed merely from the low per-query attention entropy.

Before allowing any anti-collapse KV retraining, the next step is an instrumentation-only diagnostic that leaves G9 training completely unchanged and measures hard local-slot usage on the frozen trained KV tables. If usage is healthy, G9 stands without a tuning follow-up. If actual local-slot death is severe, exactly one stronger training-only anti-collapse KV follow-up may be predeclared; no iterative tuning loop is allowed.

## Current interpretation

G9 is evidence that, under the tested one-physical-block budget, a complete nonlinear block-local transformation is a more effective payload than this resource-matched dense local KV mechanism. It is **not** a claim that nonlinear pages universally outperform Memory Layers, DSE, SCONE, PEER, or other sparse-memory systems.

The closest prior systems often activate multiple retrieved values/slots or use extra shared projections/gating. G9 tests a deliberately strict `q=1`, one-16-KiB-physical-block adaptation, not a literal reproduction of those systems.
