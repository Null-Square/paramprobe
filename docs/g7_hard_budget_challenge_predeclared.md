# G7: predeclared closest-prior-work hard-budget challenge

Status before execution: **PREDECLARED / UNRUN**

## Purpose

G7 is a kill-test for the intended ParamProbe contribution. It asks whether the hard external-parameter-probe contract is materially different from the closest published sparse-memory / retrieval architectures, and whether a strong one-probe product-key baseline can match or beat ParamProbe on the frozen larger WikiText-2 model.

G7 does **not** claim that sparse memory, product-key routing, n-gram memory, model offloading, or SSD-resident learned state are novel.

## Published mechanisms being audited

The native mechanisms are taken from the published papers, without rewriting them into ParamProbe first.

### Deep Sparse Embedding / conditional memory (Cheng et al., ACL 2026)

Source: `https://aclanthology.org/2026.acl-long.226/`.

The paper retrieves one hashed embedding from each head for each N-gram order, then concatenates all retrieved embeddings. Its DSE-27B configuration uses N-gram orders `[2,3]` and 8 hash heads, hence **16 independently indexed embedding slots per DSE module/token**. Deterministic hashing is a systems advantage and supports prefetch/offload, but the native module is not a one-slot lookup.

G7 resource classification: native DSE does not establish `q=1` under the ParamProbe logical parameter-slot accounting. It is not treated as inferior; it occupies a different point in the probe-count / payload / operator-expressivity space.

### SCONE (Yu et al., NeurIPS 2025)

Source: `https://proceedings.neurips.cc/paper_files/paper/2025/hash/2e067924aeeb02ae9919803fd08d8b4b-Abstract-Conference.html`.

SCONE stores precomputed f-gram embeddings off accelerator. For NVMe it uses LMDB/B+tree storage. The paper reports **up to four database queries per token** to find the longest matching f-gram when maximum n-gram length is 5. A single selected embedding is then used, but the published lookup procedure does not establish a worst-case one-database-probe / one-physical-block contract.

G7 resource classification: native SCONE demonstrates fixed accelerator memory/FLOPs and real NVMe feasibility, but not the same worst-case `q=1` external parameter-block guarantee.

### PEER / product-key expert retrieval (He, 2024)

Source: `https://arxiv.org/abs/2407.04153`.

PEER uses a learned query and two product-key codebooks of size `sqrt(N)` to retrieve experts from a Cartesian-product expert address space. This is the closest native competitor to ParamProbe's scalable addressing claim: it can select one expert while using sublinear resident key state and sublinear routing work.

G7 therefore includes an executable **PEER-style top-1 product-key page router** using the exact ParamProbe page operator. This is intentionally stronger than a literal PEER reproduction for the hard-block question: expert payload shape is held identical to ParamProbe so only the routing/index structure changes.

## Frozen backbone/data/assets

Use the exact archived G6a assets from workflow run `33333431198`, artifact `9738331594`:

- tokenizer SHA-256: `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`;
- backbone SHA-256: `a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049`.

Dataset remains the verified WikiText-2 raw official train/validation split, archive SHA-256 `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.

Frozen model:

- train-only byte-level BPE vocabulary 1,024;
- causal Transformer `d_model=96`, 3 layers, 4 heads, FF 384, context 128;
- insertion after block 2;
- backbone remains frozen throughout G7.

## Shared external operator and hard inference contract

Both executable G7 methods use exactly:

- `N=256` external pages;
- page operator `96 -> 20 -> 96`, tanh, residual scale 0.15;
- 3,956 FP32 parameters/page = 15,824 learned bytes;
- physical block size `B=16,384` bytes;
- top-1 selected page only (`q=1`);
- exactly 16,384 logical external parameter bytes/token;
- active selected-page matrix compute = 3,840 MACs/token.

Pages are initialized identically within each seed across the router comparison.

## Router A: ParamProbe fixed factorized hash

Identical to G6a/G6c/G6d:

- maximum address width 8 bits;
- fixed projection seed 999;
- train-only median thresholds from 40 calibration batches x 8;
- resident routing metadata: 96x8 FP32 projection + 8 FP32 thresholds = 776 scalars = 3,104 bytes;
- matrix routing compute: 96x8 = 768 MACs/token;
- no trainable router parameters.

## Router B1: PEER-style product-key, resource-matched (`d_key=6`)

A learned linear query maps 96 -> 6. Split the query into two 3-dimensional subqueries. Two learned codebooks each contain `sqrt(256)=16` 3D subkeys. The independently best subkey from each bank defines one Cartesian-product page address.

The selected product-key score is passed through a sigmoid and gates the selected page residual; this is consistent with PEER's formulation allowing nonlinear router-score activations while preserving top-1 page access.

No auxiliary load-balancing loss is added in this first kill-test. Query BatchNorm is enabled, matching PEER's published default practice for improving expert usage. BatchNorm has no learned state scaling with N.

Finite-N resource envelope:

- query matrix MACs: 96x6 = 576;
- product-key score MACs: 2x16x3 = 96;
- total router matrix/dot MACs: **672/token**, below ParamProbe's 768;
- learned query parameters: 96x6 + 6 = 582;
- learned codebook parameters: 2x16x3 = 96;
- total learned resident router parameters: **678 FP32 = 2,712 bytes**, below ParamProbe's 3,104-byte fixed metadata count.

Asymptotically, product-key codebook metadata and score work grow as `Theta(sqrt(N) * d_key)`, whereas the intended ParamProbe factorized-address family can grow logarithmically with N. G7 reports this distinction explicitly; finite-N matching does not erase it.

## Router B2: PEER-style product-key, wider (`d_key=16`)

Same algorithm, but query width 16 / subquery width 8.

Finite-N router resources:

- query MACs: 96x16 = 1,536;
- product-key score MACs: 2x16x8 = 256;
- total router MACs: 1,792/token;
- learned resident router parameters: `(96x16+16) + (2x16x8) = 1,808` FP32 = 7,232 bytes.

B2 is a quality-oriented secondary comparator, not a compute-matched baseline.

## Training protocol

Fresh page/router seeds: **32, 33, 34**.

For each seed:

- page initialization seed: `40000 + seed`, identical across all three router conditions;
- hidden-state minibatch seed: `50000 + seed`;
- materialize one frozen hidden-state bank of **1,920** minibatches x 8 sequences x 128 tokens;
- all router conditions use that exact same bank;
- page optimizer: AdamW, lr `4e-3`, weight decay `1e-4`;
- product-key router optimizer uses the same AdamW optimizer jointly with pages, same lr and weight decay;
- fixed ParamProbe router has no trainable router state.

The 1,920-step budget is chosen before execution because G6c confirmed that 256 pages need materially more sparse training exposure than the original 120-step G6a smoke budget. This is not a fixed-training-compute comparison: router training cost is reported separately.

## Evaluation

Use the exact frozen G6 validation protocol:

- official validation split;
- 40 frozen hidden-state batches x 8 sequences;
- context 128;
- evaluation RNG seed 1234.

Report for each method/seed:

- validation CE;
- normalized page-utilization entropy;
- dead-page fraction;
- selected gate mean for product-key routers;
- router resident parameters/bytes;
- router MACs/token;
- active page MACs/token;
- `q`;
- logical external bytes/token.

## Predeclared interpretation / kill criteria

This is not a winner-takes-all benchmark.

1. If **B1 (`d_key=6`) beats fixed ParamProbe in all three seeds and in mean CE** while satisfying the same `q=1`, `B=16 KiB`, active-page compute, and no-larger finite-N router metadata/MAC envelope, then the claim that ParamProbe's one-probe finite-N routing point is uniquely advantageous is **killed**. The surviving distinction is primarily asymptotic (`Theta(log N)`-style factorized address metadata/work versus product-key `Theta(sqrt N)`) and the general parameter-probe formalism.
2. If B1 does not win but B2 does, the result is a resource-quality tradeoff: product-key routing buys quality with more resident router state/work at this N.
3. If neither product-key variant wins, this supports—but does not prove—a useful quality/resource point for the current fixed factorized router.
4. Native SCONE/DSE resource classifications remain separate from the executable PEER-style quality result. Their published quality cannot be compared numerically to this tiny frozen model.
5. No outcome retroactively changes G6a/G6c/G6d/G6e.

No router hyperparameter, auxiliary loss, seed set, page shape, training duration, or pass interpretation may be changed after G7 execution begins.
