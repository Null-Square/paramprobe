# G9b execution status

Status: **IMPLEMENTED / UNRUN**

The final balanced local-KV challenge predeclared in `docs/g9b_balanced_local_kv_predeclared.md` now has an implementation and a reproducible GitHub Actions harness.

## Implementation

- Experiment: `experiments/g9b_balanced_local_kv.py`
- Experiment implementation commit: `3e8f8fa79377a2f9ebbc07f26d0a17d2924918d9`
- Workflow: `.github/workflows/g9b_balanced_local_kv.yml`
- Work branch: `g9b-balanced-local-kv`
- Parent research line: `g3b-learned-causal-replication`

The implementation keeps the frozen G9b protocol intact:

- fresh seeds 44/45/46;
- one balanced G7b `d_key=6` global router per seed, frozen and shared by both payload methods;
- 256 physical 16 KiB external blocks, `q=1`, exactly 16,384 logical external parameter bytes/token;
- unchanged `96 -> 20 -> 96` nonlinear operator baseline;
- unchanged 21-key + 21-value local-KV inference mechanism;
- fixed 327,680-state label-free Stage-A key bank from training data only;
- 600 key-only anti-collapse steps using the predeclared consistency/balance/confidence objective;
- local keys frozen permanently after Stage A;
- values-only KV task training on the exact paired 1,920-batch G9 bank;
- full G9a hard local-slot diagnostics on training and validation banks;
- predeclared anti-collapse gate and outcome classification emitted directly by the experiment.

The workflow is intentionally `workflow_dispatch` only. Creating or updating this branch therefore does **not** automatically consume GitHub Actions minutes. It pins the same CPU dependency versions and checksum-verifies the frozen G6a tokenizer/backbone and WikiText-2 archive before execution.

## Execution rule

No protocol field in the G9b predeclaration is to be changed before or after the run. After one completed execution, record the raw artifact/run provenance and classification in a separate result document. Per the predeclaration, no G9c or further KV tuning is allowed regardless of outcome.
