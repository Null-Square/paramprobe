# G7 closest-prior-work resource audit

Status: **LITERATURE/RESOURCE AUDIT COMPLETE; EXECUTABLE PRODUCT-KEY RESULT PENDING AT TIME OF THIS COMMIT**

This audit intentionally narrows the novelty claim. It asks what the closest published methods already guarantee when expressed in ParamProbe-style resource variables.

## Resource axes

The comparison separates quantities that are often conflated:

- **learned-payload probes**: independently addressed learned objects retrieved per token/module;
- **learned payload bytes**: bytes of selected learned values/operators;
- **index/metadata probes**: non-parameter lookup work needed to resolve addresses;
- **resident index/router state** and its scaling with total external capacity;
- **active compute** after retrieval;
- **physical storage evidence**: DRAM versus NVMe, cache/page behavior, and whether exact physical-byte bounds are demonstrated.

A method can have bounded learned-payload traffic while still performing multiple metadata/storage-index accesses. Conversely, many small learned lookups can have less total byte traffic than one large ParamProbe operator block.

## Deep Sparse Embedding / Conditional Memory — Cheng et al., ACL 2026

Primary source: `https://aclanthology.org/2026.acl-long.226/`.

Native mechanism:

- suffix N-grams are hashed deterministically;
- for every N-gram order `n`, each of `K` independent hash heads addresses one embedding table;
- all retrieved embeddings are concatenated;
- the DSE-27B / DSE-40B configurations use N-gram orders `[2,3]`, `K=8`, and DSE dimension 1280;
- therefore one DSE module performs **16 independently addressed learned embedding-slot lookups/token**;
- the published model places DSE modules at layers `[2,15]`, so the full architecture contains two such modules;
- deterministic hashing requires only fixed hash logic rather than an N-way learned index;
- the context-aware gate/projection/conv add active compute independent of table size.

Systems evidence:

- the paper demonstrates a 100B-parameter DSE layer resident in host DRAM;
- it reports 1.9% and 2.8% throughput penalties on 4B and 8B dense backbones in that host-memory test;
- communication volume scales with activated slots rather than total table size;
- NVMe is discussed as a future/cache tier rather than demonstrated for the 100B DSE experiment.

ParamProbe interpretation:

- DSE is already a **constant-probe / constant-selected-byte** architecture with respect to total table size if an embedding slot is treated as the learned object;
- for the published 8-head `[2,3]` module, native learned-payload `q=16` per module, not `q=1`;
- its retrieved learned object is an embedding vector, not a complete nonlinear micro-operator;
- therefore ParamProbe must not claim invention of bounded learned retrieval independent of total capacity.

## SCONE — Yu et al., NeurIPS 2025

Primary source: `https://proceedings.neurips.cc/paper_files/paper/2025/hash/2e067924aeeb02ae9919803fd08d8b4b-Abstract-Conference.html`.

Native mechanism:

- a separate f-gram model learns contextualized embeddings during training;
- at inference these outputs are precomputed into an off-accelerator f-gram embedding store;
- the main LM selects the longest frequent matching f-gram ending at each token;
- system-memory storage uses a dense matrix plus hash dictionary;
- NVMe storage uses LMDB / a B+tree mapping f-grams directly to embeddings;
- with maximum f-gram length 5, the paper reports **up to four database queries/token** to find the longest match;
- the final selected f-gram contributes one learned embedding to the token representation.

Systems evidence:

- SCONE explicitly evaluates NVMe storage;
- experiments include 10M, 100M, and 1B f-gram stores with 2048-dimensional 16-bit embeddings;
- the paper reports approximately 1.1 ms retrieval latency at batch size 1 for 10M f-grams and 2.3 ms for 1B f-grams on its NVMe setup, with lower amortized latency at larger batches.

ParamProbe interpretation:

- if `q` counts only **successfully retrieved learned embedding payloads**, SCONE can be viewed as `q=1` learned-payload retrieval/token;
- however, its published NVMe algorithm performs up to four database queries/token, and a B+tree query can itself touch multiple storage/index pages;
- the paper constrains accelerator memory/FLOPs, not a worst-case exact physical-byte or one-storage-block bound;
- its selected learned object is one embedding vector rather than one complete nonlinear operator;
- thus the meaningful distinction is physical/index probe accounting plus operator expressivity, not simply “one learned object is selected.”

## PEER — He, 2024

Primary source: `https://arxiv.org/abs/2407.04153`.

Native mechanism:

- a learned query is split into two subqueries;
- each subquery scores a codebook of `sqrt(N)` subkeys;
- Cartesian products of selected subkeys identify expert addresses;
- resident product-key metadata and score work therefore scale approximately as `Theta(sqrt(N))` for fixed key dimension, rather than `Theta(N)` for a flat router;
- the published default million-expert configuration uses 8 heads and top-16 experts/head, i.e. 128 active tiny experts, with query BatchNorm enabled to improve usage.

ParamProbe interpretation:

- native PEER already demonstrates sparse expert capacity with sublinear addressing;
- its published configuration is not `q=1`, but product-key routing can be adapted to top-1 retrieval;
- unlike DSE/SCONE, PEER retrieves nonlinear experts, making it the closest architectural competitor to page-sized ParamProbe operators;
- the G7 executable test therefore forces a PEER-style product-key router to `q=1` while holding the full ParamProbe 16 KiB page operator identical.

## ParamProbe's surviving candidate distinction after the audit

The audit **rejects** a broad novelty claim of the form:

> “learned capacity can scale while a constant number/amount of parameters is activated or retrieved.”

DSE, SCONE, PEER, PKM, and MoE literature already occupy substantial parts of that space.

The candidate contribution must instead be narrower and testable:

> A model family organized around an explicit worst-case external **parameter-block** probe contract, where a small number of independently stored blocks contain complete nonlinear micro-operators, selected learned bytes and reusable operator workspace are bounded explicitly, routing/index state is separately accounted for, and direct file-backed execution is required to match that accounting.

Within that framing, the current strongest special case is `q=1`: exactly one complete physical operator block/token.

The remaining novelty questions are:

1. Can a product-key expert router satisfy the same finite `q=1`, block size, active page compute, and resident router envelope while matching or beating ParamProbe quality? G7 executes this test.
2. Does ParamProbe's logarithmic/factorized address scaling provide a practically useful advantage over product-key `sqrt(N)` metadata/work at capacities large enough for the asymptotics to matter?
3. Does exact one-block direct-I/O execution remain competitive on characterized bare-metal NVMe once external learned stores are much larger than host cache?
4. Does storing a **complete nonlinear operator per block** yield a useful quality/byte/probe tradeoff relative to embedding-memory methods such as DSE/SCONE?

These questions—not generic sparse capacity—should determine the eventual paper claim.
