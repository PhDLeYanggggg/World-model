# OpenDD: does neighbor input improve relative trajectories?

I replayed both frozen graph backbones and the completion-trained
forecaster on the same 500 OpenDD recordings and three seeds. Each of
the seven locations is excluded from fitting its own models.
The original single-agent comparisons
reproduced. Every query defines its unordered pairs before checking
label availability. The 25,596 queries contain 5,842,092 pair-steps;
4,209,595 have labels for both agents. At least one pair-step is
labeled in 25,204 queries, the endpoint is supported in 19,482, and
all pair-steps are labeled in 3,711. Counts do not repeat seeds.

| Forecast | Available relative ADE | Available relative FDE | Query relative ADE p95 |
| --- | ---: | ---: | ---: |
| CV | 6.939 | 22.634 | 13.724 |
| OLS4 | 8.293 | 24.721 | 16.121 |
| Completion-trained | 5.376 | 20.410 | 10.661 |
| EqMotion, ego only | 5.214 | 21.205 | 10.682 |
| EqMotion, neighbors | 6.038 | 20.511 | 11.389 |
| Social-STGCNN, ego only | 6.948 | 22.485 | 13.580 |
| Social-STGCNN, neighbors | 6.187 | 21.637 | 12.193 |

| Direct paired comparison | Available gain [95% location interval] | Full-grid gain envelope |
| --- | ---: | ---: |
| EqMotion: neighbors minus ego only | -0.825 [-0.996, -0.624] | [-1.494, +0.783] |
| Social-STGCNN: neighbors minus ego only | +0.761 [+0.672, +0.867] | [-0.619, +1.712] |
| Completion-trained minus CV | +1.564 [+1.391, +1.736] | [-0.150, +2.368] |

Positive gain means smaller error. On labeled steps, observed neighbors
help Social-STGCNN but hurt EqMotion on average relative ADE. EqMotion's
relative endpoint error still improves with neighbors. The bounds do
not establish the sign of either effect over the complete twelve-step grid.
EqMotion ego-only has the best available relative ADE, whereas
the completion-trained forecaster has the best available relative FDE
and p95 here. That difference is not evidence for a release change.

Within each location, queries get equal weight. Intervals use 3,000
paired location resamples after averaging seeds. The full-grid envelope
uses all pair-steps and is not a confidence interval or necessarily a
sharp bound for pairs that share a vehicle. The readout also includes
location intervals for its endpoints. All errors are in release
coordinates. This relative-position error differs from the joint-mode
and collision metrics of [Weng et al.](https://openaccess.thecvf.com/content/ICCV2023/html/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.html).
It does not establish interaction causality, collision risk, or
independent generalization.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_backbone_relative.py
```

The run uses native CPU PyTorch, four compute threads and no worker
subprocesses. Each location-seed checkpoint and the heartbeat are local;
rerun the same command to resume. No weights were updated and no new
external outcomes were opened. The original 2% rule remains unchanged.

[Protocol](protocol.md) | [Registration](registration.json) |
[Readout](readout.json).
