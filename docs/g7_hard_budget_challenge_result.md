# G7 closest-prior-work hard-budget challenge — RESULT

Status: **COMPLETED / PREDECLARED B1 KILL CRITERION NOT MET; TOP-1 PRODUCT-KEY COLLAPSE RETAINED**

Workflow run: `33360779899`  
Workflow head SHA: `be82e8d25a775daadd832a1fa825a95e1813b784`  
Artifact: `9746861515` (`g7-hard-budget-product-key-results`)  
Artifact SHA-256: `72510c2e9599680147a5252d1affa20c5d31e9bf9434222234cfac9807d6f919`

G7 was predeclared before implementation/execution. The product-key initialization/gating details were separately frozen before implementation. No G7 router hyperparameter or auxiliary loss was changed after execution began.

## Literature/resource audit

The accompanying `docs/g7_prior_work_resource_audit.md` materially narrows the novelty claim.

- **DSE / Conditional Memory (ACL 2026):** the published 8-head bigram+trigram DSE module retrieves 16 independently addressed embedding slots/token. It is already a constant-retrieval architecture with respect to total table capacity.
- **SCONE (NeurIPS 2025):** one final f-gram embedding is used/token, while its published NVMe longest-match procedure performs up to four database queries/token. It demonstrates real NVMe embedding retrieval but not a worst-case one-physical-block byte contract.
- **MoLE (ICML 2025):** routed expert FFNs are reparameterized into storage-resident token-indexed output LUTs. Native inference loads `dN` precomputed routed-expert output parameters/token; its published 16-expert model therefore consumes all 16 output vectors/token. It is strong prior art for low-communication storage-resident neural computation results.
- **PEER (2024):** product-key routing uses sublinear `sqrt(N)` key metadata/work and retrieves nonlinear experts. Native PEER uses many tiny active experts; it is the closest architectural routing prior for the G7 executable q=1 adaptation.

Therefore ParamProbe must not claim novelty for generic bounded sparse retrieval, sparse capacity at fixed FLOPs, product-key routing, offloading, precomputed neural output lookup, or NVMe-resident learned state.

## Executable q=1 product-key challenge

Frozen shared setup:

- exact G6a tokenizer/backbone and official WikiText-2 train/validation split;
- 256 external pages;
- exact `96 -> 20 -> 96` nonlinear page operator;
- 3,956 FP32 page parameters / 15,824 learned bytes;
- physical block `B=16,384` bytes;
- `q=1` and exactly 16,384 logical external parameter bytes/token;
- active selected-page matrix compute 3,840 MACs/token;
- 1,920 page/router training minibatches;
- fresh seeds 32/33/34;
- identical page initialization and hidden-state training bank across router methods within each seed.

Router resource envelope:

| router | resident learned/fixed parameter bytes | BN buffer bytes | router matrix/dot MACs/token |
|---|---:|---:|---:|
| fixed factorized hash | 3,104 | 0 | 768 |
| product-key `d_key=6` | 2,712 | 56 | 672 |
| product-key `d_key=16` | 7,232 | 136 | 1,792 |

The `d_key=6` condition is the predeclared finite-N resource-matched adversary: it uses fewer router parameter bytes and fewer router MACs than the fixed factorized hash while satisfying the same q/B/page envelope.

## Exact results

| seed | method | validation CE | utilization entropy | dead-page fraction | gate mean |
|---:|---|---:|---:|---:|---:|
| 32 | fixed | 5.69940383 | 0.90570751 | 0.00000000 | 1.00000000 |
| 32 | pk_d6 | 5.70138509 | 0.58076569 | 0.83593750 | 0.99866903 |
| 32 | pk_d16 | 5.70086761 | 0.56765302 | 0.82421875 | 0.99974740 |
| 33 | fixed | 5.69929942 | 0.90570751 | 0.00000000 | 1.00000000 |
| 33 | pk_d6 | 5.70077386 | 0.52919019 | 0.84375000 | 0.99877232 |
| 33 | pk_d16 | 5.70109420 | 0.50434638 | 0.82421875 | 0.99996853 |
| 34 | fixed | 5.69949232 | 0.90570751 | 0.00000000 | 1.00000000 |
| 34 | pk_d6 | 5.70184387 | 0.48726223 | 0.87500000 | 0.99877489 |
| 34 | pk_d16 | 5.70207969 | 0.57819420 | 0.78515625 | 0.99961931 |

Aggregate CE:

| method | mean CE | sample std | mean delta vs fixed |
|---|---:|---:|---:|
| fixed | **5.69939853** | 0.00009656 | 0 |
| pk_d6 | 5.70133427 | 0.00053681 | +0.00193575 |
| pk_d16 | 5.70134716 | 0.00064442 | +0.00194864 |

Both product-key variants are worse than the fixed router in all three fresh seeds.

## Predeclared classification

The B1 kill condition required resource-matched `pk_d6` to beat the fixed factorized router in all three seeds and in mean CE. It does not.

`g7_b1_finite_n_uniqueness_killed=false`.

However, the result is **not evidence that PEER/product-key routing is generally inferior**. The q=1 adaptations suffer severe routing collapse: `pk_d6` leaves roughly 84–88% of pages unused in validation and `pk_d16` leaves roughly 79–82% unused. Native PEER uses multiple heads and many active tiny experts; its published BatchNorm results achieve near-complete expert usage under that different regime.

The correct interpretation is therefore:

> The current fixed factorized router occupies a better finite-N quality/resource point than this predeclared top-1 product-key adaptation, but G7 does not establish that product-key q=1 routing cannot match it. The first top-1 adaptation fails mainly through utilization collapse.

This negative baseline result is frozen. No post-result tuning of G7 is allowed.

## What survives as the candidate contribution

The literature audit and G7 jointly narrow the project to the following candidate contribution:

> **An explicit worst-case external parameter-block probe model, with complete nonlinear micro-operators stored in independently addressable physical blocks, exact selected-byte/workspace accounting, separately bounded/scalable routing state, and direct file-backed execution.**

The strongest demonstrated special case remains exactly one complete operator block/token (`q=1`). The factorized router also has a more favorable intended asymptotic metadata/work scaling than product-key `sqrt(N)` routing, but that asymptotic distinction still needs a large-capacity practical demonstration.

## Next falsification step

A single stronger q=1 product-key follow-up is justified, but it must be separately predeclared. Its purpose should be **only to remove the observed utilization-collapse confound without changing inference resources**. It must not tune against G7 validation loss. If a balanced/non-collapsed q=1 product-key adversary then beats ParamProbe under the same finite resource envelope, the finite-N routing advantage should be considered killed.
