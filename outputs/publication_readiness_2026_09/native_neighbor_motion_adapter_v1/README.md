# Local Neighbor Motion Does Not Beat Pooled Motion At Fixed Recipients

Nine new CPU PyTorch fits share the same source-separated training
cohort: 69,746 windows from nine nonempty fitting sources, excluding
all 14 development sources. The three seed-matched arms freeze the
same ungated OLS4-relative backbone and use identical initialization
and sampled rows. Each receives 900 updates at batch size 128 with
past-scale-normalized single-agent ADE. Each model allocates 5,882
parameters including the frozen backbone; absent context branches
do not contribute features or receive updates. Training took
15.4 seconds on native arm64 with two Torch threads and no loader
workers. All nine forecasts were frozen before development scoring.

| Same-recipient contrast | Observed utility (pp) | Conservative utility lower (pp), 95% CI | Observed pair ADE reduction (px) | Pair-grid lower reduction (px), 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Pooled vs. ego | +0.089 | -0.061 [-0.300, +0.099] | -0.123 | -0.510 [-0.952, -0.147] |
| Local relative vs. pooled | -0.031 | -0.194 [-0.466, -0.002] | -0.124 | -0.468 [-0.915, -0.142] |
| Local relative vs. original gain | -0.149 | -0.758 [-1.903, +0.007] | +0.023 | -1.470 [-2.970, -0.321] |

Every arm changes the same 9,675 predictions selected by the original
gain policy, with identical recipients and counts in each query.
Utility is normalized by observed CV error mass. The pair comparison
uses the same graph for all arms and 9,222,204 of 11,361,600 pair-steps
with both labels observed across the three repeated seed views.
It is a relative-coordinate error, not a collision rate or the
published joint-minimum metric of
[Weng et al.](https://openaccess.thecvf.com/content/ICCV2023/html/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.html).
The interval for the observed local-vs-pooled pair change is
[-0.299, +0.003] px; the lower-envelope comparison is not an
assertion that every possible label completion loses. Intervals
average seeds within source then resample 14 sources 3,000 times.

Ego, pooled and local pass 5/5/5 of 102 recording-seed views,
versus 4 for gain. Twenty-three easy checks are undefined in
each arm. The worst signed easy degradation is 23.98%, 71.10%
and 57.62%, respectively, all above the original 2% operating
rule. The apparent individual observed-label benefit of pooled
motion does not establish a reliable utility or pairwise benefit.
The local arm does not demonstrate the contribution of adding
relative neighbor information to this individual-ADE adapter.
Training explicitly against relative-path error is a different
hypothesis, not tested here.

[Protocol](protocol.md) | [Registration](registration.json) |
[Training receipt](train_execution.json) | [Forecast freeze](infer_execution.json) |
[Analysis](analysis.json) | [Score receipt](score_execution.json) |
[Replay receipt](replay_execution.json).

```sh
.venv-pytorch/bin/python scripts/run_m3w_neighbor_motion_adapter.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_neighbor_motion_adapter.py infer
.venv-pytorch/bin/python scripts/run_m3w_neighbor_motion_adapter.py score
.venv-pytorch/bin/python scripts/run_m3w_neighbor_motion_adapter.py replay
```

Nine focused tests passed and replay reproduces the scalar result
exactly. Data are exposed development, native eight/twelve annotation
steps and image pixels, not meters or seconds. No external access,
deployment, Stage5C or SMC.
