# G6b: training-exposure diagnostic

Status: **PROTOCOL / INFRASTRUCTURE INVALID — SCIENTIFIC TRAJECTORY NOT COUNTED**

G6b was predeclared after the frozen G6a failure to test whether the 256-page regression was caused by sparse page-training exposure. The scientific trajectory looked strongly positive, but G6b failed its own predeclared 120-step replication-integrity criterion. Therefore the result is not counted as support for the hypothesis.

## Frozen assets and resources

The workflow successfully downloaded and verified the exact G6a artifact:

- tokenizer SHA-256: `230e4b73ac91279affde8c9c62bd35bde2fdecbf0ec3d91578aa7bb1a651eedb`;
- backbone SHA-256: `a97146674da7758be0952184191076bc822e3d52fff57210facb50e00b655049`;
- WikiText-2 archive SHA-256: `ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`.

Inference resources remained frozen:

- `N=256`;
- `q=1`;
- block size 16,384 bytes;
- logical external traffic 16,384 bytes/token;
- page operator `96 -> 20 -> 96`, 15,824 learned bytes/page;
- active page matrix compute 3,840 MACs/token;
- fixed 8-factor router matrix compute 768 MACs/token;
- validation utilization entropy about 0.9057;
- dead-page fraction zero.

## Observed trajectory — exploratory only

The continuous 256-page trajectories were:

| seed | 120 | 240 | 480 | 960 | 1920 |
|---:|---:|---:|---:|---:|---:|
| 7 | 5.70341538 | 5.70193609 | 5.70070970 | 5.70001504 | 5.69957683 |
| 8 | 5.70355272 | 5.70187521 | 5.70075947 | 5.70018569 | 5.69961033 |
| 9 | 5.70339705 | 5.70195824 | 5.70080082 | 5.69996412 | 5.69950416 |

Means:

`5.70345505 / 5.70192318 / 5.70075667 / 5.70005495 / 5.69956377`.

All three checkpoint trajectories were strictly decreasing, and all three 1,920-step values were below the corresponding frozen G6a `N=16 @ 120` values.

These numbers are **not confirmatory evidence** because the predeclared integrity gate failed.

## Integrity failure

G6b predeclared that its 120-step N=256 checkpoint must reproduce the frozen G6a value within absolute CE tolerance `2e-7` for every seed.

Observed absolute errors were:

- seed 7: `6.081e-7`;
- seed 8: `2.429e-7`;
- seed 9: `7.075e-7`.

Thus all three failed the stated tolerance, despite being numerically extremely close.

The implementation also differed subtly from G6a: G6a first materialized a frozen hidden-state training bank, whereas the first G6b implementation recomputed frozen hidden states online. Cross-run CPU/multithread numerical variation can also alter factor-threshold boundary decisions at very small margins. Because the integrity tolerance was fixed before the run, it is not relaxed after observing these discrepancies.

## CI issue

The Python program correctly raised `RuntimeError` after the integrity failure. However, the workflow command was `python ... | tee ...` without `set -o pipefail`, so the shell returned `tee`'s zero exit status and GitHub Actions marked the scientific step/workflow successful.

This is an infrastructure bug. Future research workflows that pipe through `tee` must enable `pipefail` or otherwise propagate the research process exit code.

## Next confirmatory design

Rather than retuning G6b's tolerance, a separately predeclared G6c uses fresh page seeds and an **in-run paired reference**:

- train `N=16` for 120 steps;
- train `N=256` continuously through 1,920 steps;
- use the same frozen assets/router and paired minibatch prefix;
- require `N16@120 < N256@120` and then `N256@1920 < N16@120` for every fresh seed.

This tests the exposure crossover without depending on absolute CE equality across different hosted CPU runs.

## Reproducibility

- predeclaration commit: `ade21295c3990ec3acc87e7208397d76250019d5`;
- implementation commit: `6a1340715c35e84fa134584f9c7ff35692b6d7bc`;
- workflow commit: `2120ae4e88693081048bbf341572b9fa3ef67fb8`;
- workflow run: `33333782760`;
- artifact id: `9738476010`;
- artifact SHA-256: `e212d0d5763a9e675c837bbfa9cdf8feafe16e8de7e426650bc86ab925050a57`.
