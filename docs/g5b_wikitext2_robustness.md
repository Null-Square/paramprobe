# G5b: WikiText-2 robustness seeds

Status: **ROBUST CAPACITY TREND; ORIGINAL G5 GATE REMAINS FAILED**

This study was declared only after the frozen three-seed G5 result. It estimates how often small adjacent-capacity reversals occur under the unchanged WikiText-2 protocol. It must not be used to retroactively redefine or pass G5.

## Frozen protocol

Dataset and model are unchanged from G5:

- WikiText-2 raw official train/validation split;
- verified original archive: 4,721,645 bytes;
- archive SHA-256: `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`;
- byte-level frozen two-block backbone;
- internal insertion: block 1 -> ParamProbe -> block 2 -> LM head;
- page operator: `48 -> 10 -> 48`;
- 1,018 FP32 learned parameters/page = 4,072 learned bytes padded to 4,096 bytes;
- `q=1`;
- exactly 4,096 logical external parameter bytes/token;
- 960 active page matrix MACs/token;
- maximum address width fixed at 8 factors for the full `1/4/16/64/256` sweep;
- paired page-prefix initialization and paired minibatch schedules across capacities.

G5b adds ten independent page-training seeds, `10..19`, without changing any model/router/page hyperparameter. The original G5 seeds `7,8,9` and their gate outcome remain frozen.

## Additional ten seeds

### Fixed hash

| seed | 1 | 4 | 16 | 64 | 256 | strictly monotone |
|---:|---:|---:|---:|---:|---:|:---:|
| 10 | 2.48265299 | 2.48201778 | 2.48112640 | 2.48102974 | 2.48013474 | yes |
| 11 | 2.48268661 | 2.48204536 | 2.48121057 | 2.48095566 | 2.48010725 | yes |
| 12 | 2.48228100 | 2.48174780 | 2.48172035 | 2.48093967 | 2.47998220 | yes |
| 13 | 2.48303366 | 2.48203572 | 2.48162605 | 2.48095451 | 2.48011197 | yes |
| 14 | 2.48257696 | 2.48240915 | 2.48152134 | 2.48096941 | 2.48001997 | yes |
| 15 | 2.48230688 | 2.48175068 | 2.48154686 | 2.48062544 | 2.47994365 | yes |
| 16 | 2.48244140 | 2.48246611 | 2.48136206 | 2.48047709 | 2.48002306 | no |
| 17 | 2.48292124 | 2.48257056 | 2.48098859 | 2.48093675 | 2.47994911 | yes |
| 18 | 2.48209084 | 2.48190814 | 2.48074884 | 2.48050872 | 2.47996837 | yes |
| 19 | 2.48232198 | 2.48201052 | 2.48106503 | 2.48052214 | 2.47977014 | yes |

Ten-seed mean CE:

`2.48253136 / 2.48209618 / 2.48129161 / 2.48079191 / 2.48000105`.

Nine of ten new seeds are strictly monotone at every capacity step. Seed 16 has only a `1 -> 4` reversal.

### Learned prefix-balanced reliability-ordered router

| seed | 1 | 4 | 16 | 64 | 256 | strictly monotone |
|---:|---:|---:|---:|---:|---:|:---:|
| 10 | 2.48265299 | 2.48099829 | 2.47933182 | 2.47883348 | 2.47873309 | yes |
| 11 | 2.48268661 | 2.48145295 | 2.47941125 | 2.47890854 | 2.47880507 | yes |
| 12 | 2.48228100 | 2.48085441 | 2.47990088 | 2.47907188 | 2.47875320 | yes |
| 13 | 2.48303366 | 2.48149283 | 2.47957106 | 2.47939327 | 2.47891258 | yes |
| 14 | 2.48257696 | 2.48148340 | 2.48005423 | 2.47911456 | 2.47874560 | yes |
| 15 | 2.48230688 | 2.48102979 | 2.47974384 | 2.47914352 | 2.47869436 | yes |
| 16 | 2.48244140 | 2.48145113 | 2.47974319 | 2.47915266 | 2.47867181 | yes |
| 17 | 2.48292124 | 2.48142012 | 2.47971291 | 2.47885997 | 2.47860045 | yes |
| 18 | 2.48209084 | 2.48101060 | 2.47921507 | 2.47873767 | 2.47850684 | yes |
| 19 | 2.48232198 | 2.48130981 | 2.47997057 | 2.47905165 | 2.47885168 | yes |

Ten-seed mean CE:

`2.48253136 / 2.48125033 / 2.47966548 / 2.47902672 / 2.47872747`.

All ten newly declared seeds are strictly monotone through 256 pages.

## Combined 13-seed estimation

The combined set `7..19` is used only for effect estimation. It does not change the original G5 pass/fail result.

### Fixed routing

Combined mean CE:

`2.48249533 / 2.48209350 / 2.48130979 / 2.48081780 / 2.48003056`.

Stepwise improvement counts:

- `1 -> 4`: 11/13;
- `4 -> 16`: 13/13;
- `16 -> 64`: 13/13;
- `64 -> 256`: 13/13;
- `1 -> 256`: 13/13.

Eleven of thirteen seeds are strictly monotone over every step.

Paired CE changes, later minus earlier, with approximate two-sided 95% Student-t intervals:

| transition | mean delta CE | 95% interval |
|---|---:|---:|
| 1 -> 4 | -0.00040183 | [-0.00059264, -0.00021102] |
| 4 -> 16 | -0.00078371 | [-0.00107646, -0.00049096] |
| 16 -> 64 | -0.00049198 | [-0.00066450, -0.00031947] |
| 64 -> 256 | -0.00078724 | [-0.00090106, -0.00067342] |
| 1 -> 256 | -0.00246477 | [-0.00262678, -0.00230275] |

### Learned routing

Combined mean CE:

`2.48249533 / 2.48128070 / 2.47973229 / 2.47903057 / 2.47877004`.

Stepwise improvement counts:

- `1 -> 4`: 13/13;
- `4 -> 16`: 13/13;
- `16 -> 64`: 13/13;
- `64 -> 256`: 12/13;
- `1 -> 256`: 13/13.

Twelve of thirteen seeds are strictly monotone over every step. The sole learned local reversal is the already-frozen G5 seed-7 `64 -> 256` change of `+0.00002435` CE.

Paired CE changes:

| transition | mean delta CE | 95% interval |
|---|---:|---:|
| 1 -> 4 | -0.00121463 | [-0.00138766, -0.00104160] |
| 4 -> 16 | -0.00154841 | [-0.00174138, -0.00135543] |
| 16 -> 64 | -0.00070173 | [-0.00085252, -0.00055094] |
| 64 -> 256 | -0.00026053 | [-0.00036150, -0.00015955] |
| 1 -> 256 | -0.00372529 | [-0.00389656, -0.00355401] |

These intervals are descriptive paired-sample intervals over page-training seeds, not a claim that the seeds exhaust all sources of experimental uncertainty.

## Interpretation

The original G5 strict three-seed gates remain failed and are not reclassified.

The larger robustness study supports a stronger, but still bounded, statement:

> On WikiText-2 raw under the frozen one-page traffic protocol, the endpoint capacity benefit is sign-consistent across all 13 evaluated page-training seeds. Learned routing is strictly monotone through all five capacity points in 12/13 seeds, and its mean `64 -> 256` change is negative with a paired 95% interval separated from zero.

Thus the isolated G5 learned seed-7 `64 -> 256` reversal is better interpreted as a small stochastic local reversal than as evidence for a universal 256-page learned-routing wall on this corpus. The Tiny Shakespeare 256-page plateau remains a separate frozen constraining result.

## Reproducibility

- G5 original workflow run: `33329027660`;
- G5 original artifact id: `9737125662`;
- G5 original artifact SHA-256: `20cfac3e12a7413e08da9b8d0ebdcfa0da804f6c3996657720a7f2c1b1670360`;
- G5b robustness workflow run: `33329288729`;
- G5b artifact id: `9737264713`;
- G5b artifact SHA-256: `099c964ed9652ec364047b6e8451e017e20a58fdd3d24a938a455dd7951d99e6`;
- environment: Python 3.12.14, NumPy 2.3.5, PyTorch 2.10.0+cpu, Ubuntu 24.04.
