# G6c: paired training-exposure crossover result

Status: **PASSED / TRAINING-EXPOSURE CROSSOVER CONFIRMED**

G6c was predeclared before execution in `docs/g6c_paired_exposure_crossover_predeclared.md`. It uses fresh page seeds 10/11/12, the exact archived G6a tokenizer/backbone, the same fixed factorized router, the same 16 KiB page operator, and paired training-data prefixes. All decisive comparisons are made within the same workflow run.

## Frozen assets and inference resources

- tokenizer SHA-256: `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`
- backbone SHA-256: `a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049`
- WikiText-2 raw archive SHA-256: `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`
- `q=1`
- block bytes: 16,384
- logical external parameter bytes/token: 16,384
- page operator: `96 -> 20 -> 96`
- learned page payload: 15,824 bytes in one 16 KiB block
- active page matrix MACs/token: 3,840
- fixed-router matrix MACs/token: 768
- N=16 utilization entropy: 0.92041132; dead-page fraction: 0
- N=256 utilization entropy: 0.90570751; dead-page fraction: 0

Inference resources do not change with page-training duration.

## Fresh-seed paired results

| seed | N=16 @ 120 | N=256 @ 120 | N=256 @ 1920 | 16 beats low-exposure 256 | 256 crossover at equal mean exposure/page |
|---:|---:|---:|---:|:---:|:---:|
| 10 | 5.70218958 | 5.70326931 | 5.69937409 | yes | yes |
| 11 | 5.70235314 | 5.70332602 | 5.69955056 | yes | yes |
| 12 | 5.70200230 | 5.70325141 | 5.69923952 | yes | yes |

The three-seed N=16 reference mean is `5.70218167`.

N=256 training trajectory:

| steps | mean routed assignments/page | validation CE mean | sample std |
|---:|---:|---:|---:|
| 120 | 480 | 5.70328225 | 0.00003895 |
| 240 | 960 | 5.70176050 | 0.00006504 |
| 480 | 1,920 | 5.70061757 | 0.00019843 |
| 960 | 3,840 | 5.69984304 | 0.00025267 |
| 1,920 | 7,680 | 5.69938806 | 0.00015599 |

At the final checkpoint, N=256 and N=16 have the same **mean routed training assignments/page** (7,680), while N=256 uses 16x more total routed page-training assignments/optimization work.

Mean final N=256 improves over the paired N=16 reference by `-0.00279361` CE and improves over its own low-exposure 120-step point by `-0.00389419` CE.

All three fresh seeds satisfy every predeclared primary condition:

- low-exposure direction: N=16 @ 120 beats N=256 @ 120;
- equal-per-page-exposure crossover: N=256 @ 1920 beats N=16 @ 120;
- within-N improvement: N=256 @ 1920 beats N=256 @ 120;
- frozen inference-resource assertions hold;
- additionally, every N=256 checkpoint improves monotonically in all three seeds.

Therefore `g6c_exposure_crossover_confirmed=true`.

## Interpretation

G6c confirms that sparse page training exposure/optimization is a **major cause** of the G6a 256-page failure. It does not make G6a pass: G6a remains the frozen result showing that 256-page scaling fails when total page-training steps are held fixed at 120.

The correct distinction is:

> Inference resource scaling can remain bounded while training resources required to realize the inactive external capacity grow with N.

G6c does not establish an optimal training-scaling law and does not show capacity gains at fixed total training compute. It motivates a new capacity sweep with an explicit, predeclared capacity-dependent page-training budget recorded as a separate resource.

## Reproducibility

- workflow run: `33334243338`
- artifact: `g6c-paired-exposure-crossover-results`
- artifact id: `9738601733`
- artifact SHA-256: `4274f7e4011bbb65acdacb035905bcadb0c169cc51529dc749599966d0c246e8`
- workflow head SHA: `aca449fd324267aba4218ce16b752ed1643d2d8e`
