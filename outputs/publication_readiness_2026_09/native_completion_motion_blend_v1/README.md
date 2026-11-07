# Missing-Label Debits Shrink The Learned Pair Correction

This matched test freezes the two previously trained individual and
pair forecasts and learns only a scalar per-agent blend. The six
new fits use the original 69,746 training rows, nine source groups,
and same-query graph, excluding all exposed development sources.
Both blend heads start at half interpolation and use identical
source-balanced single and paired draws for 900 updates. Only one
penalizes disagreement on wholly unobserved individual rows and
unobserved pair steps. This uses the same uncertainty quantities
as the evaluator, but the scaled batch loss is a training surrogate,
not the exact recording-level deployment test.

| Pair ADE contrast vs frozen individual reference | Co-observed reduction (px) | Full-grid lower reduction (px), 95% CI |
| --- | ---: | ---: |
| Full correction (previous matched experiment) | +0.0440 | -0.2385 [-0.6069, -0.0223] |
| Fixed half correction | +0.0234 | -0.1185 [-0.3026, -0.0106] |
| Learned on observed labels | +0.0163 | -0.0772 [-0.1875, -0.0093] |
| Learned with missing debits | +0.0024 | -0.0039 [-0.0080, -0.0006] |

All arms use the same original 9,675 switched people and query
counts. Three forecaster seeds are averaged before the 3,000
paired source bootstrap draws over the 14 development groups.
The observed and full-grid metrics use different support and
denominators. A negative lower comparison does not prove that
every completion loses. The treatment's point increase in
conservative individual utility vs frozen individual is only
+0.0011 pp [-0.0023, +0.0046]. Directly versus the observed-only
blend, its pair lower contrast is -0.0902 px [-0.2006, -0.0202],
not evidence that the learned treatment is better.

On the matched substitutions, the observed-only alpha
10th/50th/90th percentiles are 0.281/0.412/0.692; adding the
missing debits reduces these to 0.017/0.058/0.248. The learned
policy largely suppresses the pair correction instead of finding
a reliable way to deploy it. All four compared replacement
arms pass 5/102 recording-seed checks; the original gain
policy passes 4/102. Worst signed easy degradation is 65.55%
for the treatment, versus 65.52% for the individual reference.
Neither passes the original 2% operating rule. No independent
evidence or promotion.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_completion_motion_blend.py train --resume
.venv-pytorch/bin/python scripts/run_m3w_completion_motion_blend.py infer
.venv-pytorch/bin/python scripts/run_m3w_completion_motion_blend.py score
.venv-pytorch/bin/python scripts/run_m3w_completion_motion_blend.py replay
.venv-pytorch/bin/python scripts/plot_m3w_completion_motion_blend.py
```

Training and fitting-source anchor reconstruction took 13.2 s on
native arm64 with two Torch threads and no loader workers. Nine
focused tests passed. Scores replay exactly and the vector figure
regenerates byte for byte from the two hash-verified analyses:
`0d06de1c6366806755cc9bf9d6cebea02806433ffd08938548956b7df3f03cff`.

[Protocol](protocol.md) | [Registration](registration.json) |
[Training](train_execution.json) | [Forecast freeze](infer_execution.json) |
[Analysis](analysis.json) | [Score](score_execution.json) |
[Replay](replay_execution.json). Exposed development only, in image
pixels and native eight/twelve annotation steps. No external access,
deployment, Stage5C or SMC.
