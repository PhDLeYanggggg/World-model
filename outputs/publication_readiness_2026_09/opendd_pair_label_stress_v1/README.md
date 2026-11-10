# OpenDD: the same label coverage can give a different answer

I used 3,711 originally complete queries from the frozen OpenDD
backbone comparison. They contain 242,088 pair-steps before masking.
All forecasts and targets stay fixed. Hiding the first three, last
three or three uniformly chosen steps leaves exactly 181,566
pair-steps in every arm. The six-step sensitivity keeps 121,044.

| Relative ADE gain | All 12 steps | Hide first 3 | Hide last 3 | Uniformly hide 3 | First-minus-last [95% location interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| EqMotion: neighbors vs. ego only | -0.209 | +0.035 | -0.470 | -0.209 | +0.505 [+0.401, +0.625] |
| Social-STGCNN: neighbors vs. ego only | +1.134 | +1.348 | +1.108 | +1.135 | +0.240 [+0.197, +0.280] |
| Completion-trained vs. CV | +2.243 | +2.767 | +2.047 | +2.245 | +0.720 [+0.658, +0.784] |

Positive gain means lower error. For EqMotion the available-label
point estimate flips sign when the first three steps are hidden,
but its 95% location interval [-0.075, +0.131] does not establish
improvement. The complete-label interval [-0.279, -0.138] favors
ego only. At 50% coverage, first versus last masking gives +0.276
versus -0.693 for the same forecasts. With uniformly placed masks,
mean bias over the 32 replicates is within 0.003 release-coordinate
units of zero for all three comparisons at both severities.

The deterministic envelopes cover the known gain for every mask.
At 75% coverage, the Social-STGCNN full-grid lower endpoint is
+0.732 after hiding the first three steps, but -0.019 after hiding
the last three. Thus even sign certification depends on where labels
are absent. These envelopes are not confidence intervals. The paired
intervals average seeds within each location and resample seven
locations 3,000 times. The 32 uniform masks are not independent
recordings.

This is a controlled removal of labels from an already exposed
complete-query subset. It does not show that natural missingness has
this pattern or magnitude. No forecasts, thresholds, independent
outcomes or release decisions changed.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_pair_label_stress.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_relative_label_stress.py tests/test_m3w_relative_forecast_readout.py tests/test_m3w_two_factor_bounds.py tests/test_m3w_full_grid_risk.py
```

The audit reproduces original individual forecasts from verified
checkpoints, then saves a scalar receipt per location and seed.
Repeat the same command to resume. No training or CREATE job was needed.

[Protocol](protocol.md) | [Registration](registration.json) |
[Readout](readout.json).
