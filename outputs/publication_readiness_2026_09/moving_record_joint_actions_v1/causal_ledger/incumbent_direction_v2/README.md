# Fixed-Recipient Direction Changes on Pedestrian Development

## Why the geometry policy lost

The [observability study](../observability_v1/README.md) reduced edits on
unlabeled rows at identical query counts, but lost conservative utility.
I replayed its frozen predictions and split the paired effect into measured
gain and the disagreement radius on wholly unobserved rows, using the same
full-reference denominator for every part. All three matched contrasts
reproduce their registered value for every source and seed.

| Geometry versus original screen | Observed pp | Disagreement debit pp | Conservative pp |
| --- | ---: | ---: | ---: |
| All matched rows | -0.195 | 1.439 | -1.634 |
| Original recipient removed | -0.220 | 0.683 | -0.903 |
| New recipient, admitted | +0.092 | 0.029 | +0.062 |
| New recipient, forced fill | -0.032 | 0.501 | -0.533 |
| Direction change, admitted | -0.004 | 0.167 | -0.170 |
| Direction change, forced fill | -0.031 | 0.058 | -0.089 |

Dropping all 4,770 fills alone leaves a loss of -1.344 pp
[-2.457, -0.413]. This diagnostic no longer matches edit count.
The debit is uncertainty in the paired comparison, not observed harm.
An unlabeled track edited by only one policy can contribute even when
the new policy edits fewer unlabeled tracks in total.
[Attribution](../observability_v1/cost_attribution_v1/analysis.json).

## Direction experiment

Keep every agent selected by the original screen. A small Torch head
learns signed ADE gain for two candidate paths on labeled fitting
rows, reversing sign if their order is reversed. At inference, use
the frozen geometry missingness probability to weight the **pairwise**
path-disagreement radius, and allow at most one positive predicted
direction swap per query. Absolute-cost subtraction and the no-debit
control use the same query set for swaps. This is a development
comparison, not a calibrated safety certificate. It follows the general
idea of predicting for a downstream decision rather than comparing
prediction accuracy in isolation; see
[Mandi et al., ICML 2022](https://proceedings.mlr.press/v162/mandi22a.html).

| Rule | Swap queries | Joint passes / 102 | Risk passes | Lower utility versus original, pp [95% interval] |
| --- | ---: | ---: | ---: | --- |
| Original screen | 0 | 1 | 2 | 0 |
| Paired gain | 2,149 | 0 | 1 | -0.113 [-0.232, -0.021] |
| Absolute-cost subtraction | 2,149 | 0 | 1 | -0.093 [-0.186, -0.012] |
| Paired gain, no debit | 2,149 | 0 | 1 | -0.269 [-0.470, -0.101] |
| Outcome-aware maximum utility, diagnostic | 3,328 | 2 | 6 | +0.679 [+0.355, +1.063] |

The paired head loses to absolute-cost subtraction by -0.173 pp
[-0.461, -0.002] in the direct comparison. The direct paired contrast
is not the difference of the two lower bounds in the table: each
contrast debits its own exact path disagreements. All rows retain
12,235 edits, 377 completely unlabeled edits, and 29 undefined-risk
views. The outcome-aware arm maximizes the specified utility, not the
number of joint passes; it is not an upper bound on risk-feasible passes.
Intervals use 3,000 paired source resamples (14 sources, 34 recordings,
three seeds; 102 repeated views).

The first launch stopped at fold two: one allowed training source had
no known alternatives after recipient locking. The registered v2
samples only allowed donors with usable pairs. Of 42 folds, 27 had
all seven usable donors; the others had between two and six. Held
and reserved roles were not changed. The experiment narrows the
failure: removing recipient churn helps limit disagreement, but fixed
recipients also preserve missing risk support. No policy was promoted.
Before another direction refit, inspect whether **changing support
recipients** can increase easy-reference coverage with a small
disagreement radius. Require positive matched utility and the existing
recording safety checks together. The 30 DroneCrowd confirmation
groups remain sealed. No Stage5C, SMC, or new metric/seconds claim.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/audit_m3w_moving_observability_cost.py
.venv-pytorch/bin/python scripts/run_m3w_incumbent_direction.py
.venv-pytorch/bin/python scripts/run_m3w_incumbent_direction.py --replay
.venv-pytorch/bin/python scripts/build_m3w_incumbent_direction_replay.py
```

`fresh_run`: paired attribution and 42 fits, 21,000 optimizer updates,
41.656 seconds, PID 86274, four arm64 Torch threads, zero loader workers.
Every 100 steps: checkpoint and heartbeat; repeat the command to resume.
`cached_verified`: frozen candidate and policy parents and zero-update
replay. `not_run`: independent confirmation or new CREATE training.
