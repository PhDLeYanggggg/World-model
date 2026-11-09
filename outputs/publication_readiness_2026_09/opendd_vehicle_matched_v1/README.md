# The vehicle policy abstains because of the predicted easy risk

The [vehicle forecaster](../opendd_vehicle_residual_v1/README.md) improves
on CV before selection, but almost always fails the original risk rule
when used on every actor. I kept that forecaster fixed for each held
location and tested whether learning when to edit preserves its gains.
This is a development experiment on OpenDD, not Songdo confirmation.

## Forecasts used to train the selector

For a selector-training action on site B while site A is held, its
neural candidate is predicted by a model fitted on neither A nor B.
I trained 63 such predictors and 21 gain/harm models, with three seeds
and 31,500 optimizer updates. The outer prediction still uses the
previous frozen model trained on the six donor sites, not a retrained
held-site model. The value training inputs contain only past histories
and candidate paths; future coordinates provide supervision only.

Each query has the same CV/OLS4/median4/neural bank and the same
shortening-only radius slots in all arms. The joint rule predicts gain
against both references and sums estimated all/easy risk excess before
selecting up to two edits. Independent and common-family controls match
its exact realized counts and radii. I froze the predictions at
`9a872ba32` before scoring. No held outcomes were used to fit a threshold.

| Frozen policy | Edits | CV conservative gain | Original 2% risk passes |
| --- | ---: | ---: | ---: |
| Joint | 0 | 0.0 pp | 0 / 1,500 |
| Matched independent | 0 | 0.0 pp | 0 / 1,500 |
| Matched family | 0 | 0.0 pp | 0 / 1,500 |

All three fall back to CV. No active recording supports a paired
utility or interaction comparison; both intervals are **not estimable**.
The zero numerator is not evidence of a safety pass: the original
selected-risk ratio is undefined at zero interventions.

## Did the forecast menu or the risk estimate fail?

| Held candidate condition | Count |
| --- | ---: |
| Legal equal-radius action instances across seeds | 1,187,471 |
| Predicted positive incremental gain against both references | 272,696 |
| Predicted positive gain and acceptable all-risk excess | 112,029 |
| Predicted acceptable easy-risk excess | **0** |
| One-action hindsight witnesses verified by the original scorer | **1,429 / 1,500 recording-seed views** |

The hindsight test makes at most one edit per recording and requires a
nonzero true easy reference, positive conservative gain, and the
unchanged risk tests. It is a proof of available opportunities, not a
selector or an upper bound on multi-edit performance. Even the chosen
single edit per recording yields only about +0.048 pp on the average
site view. Candidate instances share windows, and the same 500
recordings appear at three seeds; these are not independent samples.

The bottleneck is the **predicted** easy constraint, not lack of
predicted gain. Across action instances, mean predicted easy positive
harm plus unknown-label debit is 0.0759 in native units, versus 0.0352
in the evaluated targets. The mean predicted easy reference mass is
0.408 versus 0.568. These are pooled component means, not per-recording
risk ratios. Both biases move the estimated easy constraint in the
direction of abstention. Since every predicted easy excess is positive,
adding a second action cannot make it feasible by summation.

**Decision:** Keep CV fallback. Do not relax 2% or search a larger joint
menu with this score. The next useful comparison is a two-part easy
model that estimates usable easy-reference support separately from
conditional harm and calibrates on a source excluded from both the
forecaster and selector. Compare equal-count utility only once there
are actual edits. Neither Songdo nor sealed confirmation data is needed
to diagnose the current regression failure.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_vehicle_matched.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_vehicle_matched.py score
.venv-pytorch/bin/python scripts/diagnose_m3w_vehicle_matched.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_matched.py tests/test_m3w_vehicle_residual.py
```

`fresh_run`: 84 native arm64 Torch fits, frozen choice evaluation and
the one-edit hindsight diagnostic. `cached_verified`: parent vehicle
cohorts and outer checkpoint hashes. Four compute threads, no data
workers, checkpoints every 100 steps and a PID heartbeat. Repeating the
commands resumes training or verifies the readout. No new threshold,
external scoring, deployment, Stage5C or SMC.
[Protocol](protocol.md) | [Freeze](freeze.json) | [Readout](readout.json) |
[Hindsight diagnosis](diagnosis.json).
