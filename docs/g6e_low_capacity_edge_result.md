# G6e: paired low-capacity edge robustness result

Status: **LOW-CAPACITY EDGE UNRESOLVED**

G6e was predeclared after the frozen G6d failure and before implementation/execution. It is an estimation study only; it does not alter G6d's predeclared pass/fail outcome.

## Frozen protocol

The study reuses the exact archived G6a tokenizer/backbone and the G6d fixed-router/page configuration:

- tokenizer SHA-256: `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`;
- backbone SHA-256: `a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049`;
- WikiText-2 raw archive SHA-256: `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`;
- `q=1`;
- physical block and logical external parameter traffic: 16,384 bytes/token;
- page operator `96 -> 20 -> 96`, 3,956 FP32 parameters / 15,824 learned bytes;
- active page matrix MACs/token: 3,840;
- fixed maximum-width router matrix MACs/token: 768;
- both N=4 and N=16 train for exactly 120 steps / 122,880 routed token assignments;
- fresh paired page seeds 16..31;
- identical hidden-state/target training bank within each seed for both capacities;
- identical frozen validation hidden-state bank for all conditions.

Mean routed assignments/page are 30,720 for N=4 and 7,680 for N=16, exactly as in G6d.

## Exact paired results

| seed | N=4 CE | N=16 CE | CE16 - CE4 | N=16 better |
|---:|---:|---:|---:|:---:|
| 16 | 5.70213271 | 5.70230093 | +0.00016822 | no |
| 17 | 5.70232022 | 5.70242301 | +0.00010279 | no |
| 18 | 5.70243658 | 5.70222454 | -0.00021204 | yes |
| 19 | 5.70246321 | 5.70229435 | -0.00016886 | yes |
| 20 | 5.70246550 | 5.70249524 | +0.00002974 | no |
| 21 | 5.70217565 | 5.70228748 | +0.00011183 | no |
| 22 | 5.70231087 | 5.70244417 | +0.00013330 | no |
| 23 | 5.70213715 | 5.70235429 | +0.00021714 | no |
| 24 | 5.70238225 | 5.70241526 | +0.00003301 | no |
| 25 | 5.70215567 | 5.70228804 | +0.00013237 | no |
| 26 | 5.70209321 | 5.70233457 | +0.00024136 | no |
| 27 | 5.70253261 | 5.70235181 | -0.00018080 | yes |
| 28 | 5.70223953 | 5.70222919 | -0.00001034 | yes |
| 29 | 5.70232083 | 5.70248456 | +0.00016373 | no |
| 30 | 5.70219932 | 5.70225536 | +0.00005604 | no |
| 31 | 5.70236962 | 5.70232741 | -0.00004221 | yes |

Aggregate estimates:

- N=4 mean CE: `5.70229593`;
- N=16 mean CE: `5.70234439`;
- paired mean `CE16 - CE4`: `+0.00004846`;
- paired sample standard deviation: `0.00014011`;
- paired standard error: `0.00003503`;
- two-sided 95% Student-t interval: `[-0.00002621, +0.00012312]`;
- N=16 lower-loss in `5/16` fresh seeds.

The predeclared classification rule therefore yields:

`g6e_edge_classification=unresolved`.

The interval overlaps zero, so this study does not support a reliable N=4 -> N=16 improvement or regression under the frozen 120-step protocol.

## Interpretation

The small G6d mean reversal at `4 -> 16` is not established as a stable capacity effect. With 16 additional paired seeds, the effect remains near zero relative to seed-to-seed page-training variation.

This does **not** reclassify G6d: its strict mean-monotonicity gate remains failed exactly as predeclared.

Combined G6 evidence now supports a narrower and more defensible statement:

> At the larger subword-LM scale, the low-capacity N=4 versus N=16 edge is effectively unresolved under the frozen short training budget, while the N=16 -> N=64 -> N=256 regime shows much larger improvements once an explicit per-page training-exposure floor is supplied. Inference traffic and active page compute remain fixed; total page-training work does not.

Further seed-chasing on the N=4/N=16 edge is not a high-value next experiment. The more important open questions are the training-efficiency law, learned-routing behavior at this scale, and physical quality/latency tradeoffs on characterized hardware.

## Reproducibility

- workflow run: `33335038289`;
- artifact: `g6e-low-capacity-edge-results`;
- artifact id: `9738795620`;
- artifact SHA-256: `b5c3c8e45e732311167053d1c30d3b42f3f78baa239dd2006c3340b5d5b75b7a`;
- workflow head SHA: `8c476066368c01e8cb340e99daf51739815794a3`.
