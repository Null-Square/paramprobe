# G9a result: hard local-KV usage diagnostic

Status: **COMPLETED — LOCAL KV USAGE UNHEALTHY; ONE G9b FOLLOW-UP AUTHORIZED**

G9a was predeclared in `docs/g9a_local_kv_usage_diagnostic_predeclared.md` before execution. It changes no G9 training rule or inference resource. It reproduces G9 and adds hard local-slot usage instrumentation only.

## Provenance

The execution used the exact archived G6a assets and checksum-pinned WikiText-2 raw data:

- tokenizer SHA-256: `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`;
- backbone SHA-256: `a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049`;
- G6a artifact SHA-256: `0e901efe5d360b905fbb15c5cdb5d90ffea3ad60d7f05e2712fe43fc37d2ad21`;
- WikiText-2 archive SHA-256: `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.

G9a ran on the isolated private Actions runner harness because the ParamProbe organization repository had recent zero-step runner-provisioning failures. This changes execution location only, not research protocol.

Workflow run: `33539526688`  
Job: `99962025480`  
Artifact: `paramprobe-g9a-usage-results`  
Artifact id: `9813397114`  
Artifact SHA-256: `98972c8683fb18a8e118670ff4f5c7719f99fb68954f5ee087ea625baeb19aed`

## G9 reproduction check

The unchanged G9 training path reproduced the frozen G9 CE values to only a few `1e-6` CE of floating-point drift:

| seed | operator CE | delta vs frozen G9 | KV CE | delta vs frozen G9 |
|---:|---:|---:|---:|---:|
| 38 | 5.69769385 | -3.47e-6 | 5.70200177 | +1.57e-6 |
| 39 | 5.69791871 | -6.55e-6 | 5.70183392 | +4.66e-7 |
| 40 | 5.69873184 | -1.91e-6 | 5.70246342 | -2.16e-7 |

Thus G9a is a faithful diagnostic rerun. The frozen G9 classification remains `nonlinear_operator_supported`.

## Hard local-slot usage

Each KV block contains 21 local slots, so the 256 global blocks contain `256 * 21 = 5,376` `(block, local-slot)` pairs. G9a evaluates hard local slot `argmax_j <x,k_j>` on the same 1,920-batch page-training bank: 1,966,080 routed token queries per seed.

Predeclared health gate:

- normalized global hard local-slot usage entropy >= 0.85; and
- dead local-slot fraction <= 0.05;
- both must hold in every seed.

### Training-bank results

| seed | queries | global hard-usage entropy | dead local-slot fraction | active local slots | weighted per-block entropy | median active fraction/block | median max slot share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 38 | 1,966,080 | 0.62155796 | 0.95238095 | 256 | 0.00000000 | 0.04761905 | 1.00000000 |
| 39 | 1,966,080 | 0.59387816 | 0.95238095 | 256 | 0.00000000 | 0.04761905 | 1.00000000 |
| 40 | 1,966,080 | 0.60898286 | 0.95238095 | 256 | 0.00000000 | 0.04761905 | 1.00000000 |

The 10th-percentile active fraction is also `1/21` in every seed and the 90th-percentile maximum local-slot traffic share is 1.0.

This is genuine local-memory collapse: **exactly one local slot per observed global block is ever selected**. Only 256 of 5,376 local slots are active and 95.238095% are dead.

Validation hard usage is consistent with the training diagnosis. Seeds 38/39 use 256 local slots; seed 40 uses 247 because the shared global router itself has several validation-dead global blocks.

The original near-zero per-query local-softmax entropy is therefore now correctly interpreted as both sharp retrieval and severe local-slot death in this G9 KV training protocol.

## Predeclared classification

`local_usage_healthy_by_seed=38:false,39:false,40:false`  
`local_usage_healthy=false`  
`g9a_classification=usage_unhealthy_g9b_allowed`  
`g9b_allowed=true`

G9a does **not** retroactively change the G9 result: the nonlinear operator still beat the exact resource-matched KV mechanism that was tested. However, the payload-level claim should not be treated as closed until the one pre-authorized stronger KV anti-collapse protocol G9b is executed.

Exactly one G9b training-only anti-collapse follow-up is now allowed. There will be no G9c or iterative KV tuning after G9b, regardless of outcome.