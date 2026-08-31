# G8: address-scaling crossover local implementation result

Status: **EXACT RESOURCE CLAIM CONFIRMED LOCALLY; CI/WORKFLOW REPRODUCTION PENDING**

The predeclared GitHub Actions workflow run `33371240765` failed twice before recording any job step or log. Therefore those attempts are classified as infrastructure failures and provide no scientific result.

The already-committed G8 implementation was then executed unchanged in the local CPU research environment for an implementation cross-check. Exact analytical resource counts are independent of hardware/timing; CPU timing remains diagnostic only.

Local environment:

- PyTorch `2.10.0+cpu`;
- Python `3.13.5` (different from the intended CI Python 3.12; irrelevant to exact counts, relevant to timing portability);
- `torch.set_num_threads(8)` as predeclared.

## Exact resource table

| pages N | logical external capacity at 16 KiB/page | factor MACs/token | product-key MACs/token | factor resident bytes | product-key resident bytes incl. BN |
|---:|---:|---:|---:|---:|---:|
| 256 | 4 MiB | 768 | **672** | 3,104 | **2,768** |
| 4,096 | 64 MiB | 1,152 | **960** | 4,656 | **3,920** |
| 16,384 | 256 MiB | 1,344 | 1,344 | **5,432** | 5,456 |
| 65,536 | 1 GiB | **1,536** | 2,112 | **6,208** | 8,528 |
| 1,048,576 | 16 GiB | **1,920** | 6,720 | **7,760** | 26,960 |
| 16,777,216 | 256 GiB | **2,304** | 25,152 | **9,312** | 100,688 |
| 1,073,741,824 | 16 TiB | **2,880** | 197,184 | **11,640** | 788,816 |

The predeclared exact resource gate is satisfied: at every point N >= 65,536, the factorized router uses strictly fewer routing MACs/token and fewer resident routing bytes than the exact two-bank d_key=6 product-key router.

## Local timing diagnostic

Same 512 x 96 deterministic hidden batch; 5 warmups; 20 timed iterations; 8 PyTorch threads. Values are median nanoseconds/token.

| pages N | factor median ns/token | product-key median ns/token |
|---:|---:|---:|
| 256 | 528 | 1,022 |
| 4,096 | 624 | 1,318 |
| 16,384 | 810 | 1,746 |
| 65,536 | 770 | 1,520 |
| 1,048,576 | 679 | 2,512 |
| 16,777,216 | 843 | 11,164 |
| 1,073,741,824 | 976 | 73,513 |

These timings are implementation/hardware diagnostics only and do not determine G8 pass/fail. The fact that factorized routing is faster even at small N despite somewhat larger exact MAC count reflects kernel/matrix-shape effects and should not be generalized to other devices.

## Interpretation

G7b and G8 together give a clean tradeoff:

- at N=256, balanced product-key routing is better LM quality and cheaper finite-N routing;
- around N=16k pages / 256 MiB logical external capacity, the exact routing-resource curves cross;
- beyond that, the current binary factorized address family becomes increasingly cheaper in resident state and exact score/matrix work;
- G8 does **not** establish that factorized routing retains acceptable quality/utilization at these much larger N values.

The next architecture question should therefore not be another small-N router contest. It should test whether the **complete nonlinear operator stored in one block** offers a useful quality-per-byte/probe advantage over an embedding/key-value memory block under the same q=1 physical contract. This is orthogonal to G7b's routing result.

A CI rerun of the exact same G8 code remains desirable when GitHub Actions runner provisioning is available; no protocol change is needed.