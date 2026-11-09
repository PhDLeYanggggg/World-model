# DroneCrowd: Predicting Benefit Separately from Harm

The [history risk experiment](../dronecrowd_recording_history_risk_v1/README.md)
showed that observed motion is useful for predicting harm and that the
independent forecast outperforms the risk-priced menu on matched views.
Here I keep that independent forecast fixed and ask whether the same
history can predict when it is worth releasing.

## Design

I fit each analytic path on the first four of the eight input positions.
On the last four observed positions, I measure OLS4 error minus the error
of the frozen independent action identity. The feature aggregates query
means and normalizes by observed CV error. It does not see the twelve
future positions. The regression target is twelve-step conservative
gain over OLS4, including the existing missing-label debit, in units
of CV-normalized percentage points. The actual release check still
requires improvement over both CV and OLS4 and the original two 2%
risk limits with support.

I fit a fixed ridge per held source on the other fourteen sources with
equal source weight. The predictors use an intercept alone, the donor
gain score, the history feature, or both. A lower estimate subtracts
the worst overprediction on nested held-source folds. The risk point
and log-envelope predictions are unchanged from the previous study.
The primary contrast is combined versus donor ridge, with 3,000 paired
source bootstrap draws. This is related to [learning to defer](https://proceedings.mlr.press/v119/mozannar20b.html),
but the comparison here is between frozen trajectory forecasts under
separate recording-level gain and harm constraints, not expert
classification error.

The experiment uses 40 previously opened calibration recordings from
15 sources, three frozen seeds and 120 recording-seed views. The feature
pass reads only histories. The evaluation reads previously computed
score summaries. No training, threshold or prediction uses the labels
of its outer held source; the development groups themselves are not
new independent evidence. Coordinates are native image units; the
protocol uses eight observed and twelve forecast annotation steps.

## Results

| Utility input | Held-source MAE (pp) | Source-averaged sign accuracy |
|---|---:|---:|
| Raw donor score | 1.671 | 63.0% |
| Intercept alone | 0.398 | 63.0% |
| Calibrated donor | 0.338 | 56.5% |
| Calibrated history | 0.145 | 89.4% |
| Calibrated donor + history | 0.146 | 87.9% |

The primary gain-MAE reduction is **0.192 pp [0.127, 0.259]**.
History alone reduces it by **0.193 pp [0.128, 0.261]**. Neither
point estimate changes which views are released relative to the raw
donor gain sign. The lower utility estimate changes the release set:

| Risk and utility check | Selected views | Actual risk pass | Both accuracy floors and risk |
|---|---:|---:|---:|
| Risk point + raw donor gain | 23 | 22 | 22 |
| Combined lower utility alone | 11 | 9 | 9 |
| Risk point + history lower utility | 7 | 7 | 7 |
| Risk point + combined lower utility | 8 | 8 | 8 |
| Log-risk envelope + raw donor gain | 7 | 7 | 7 |
| Log-risk envelope + combined lower utility | 3 | 3 | 3 |

The risk point and combined lower utility select eight views from
three source groups, with conservative gains of +0.580, +0.751 and
+0.745 pp over OLS4 in those selected subsets. Without the risk
check the same utility rule would select two failing views. In the
other direction, adding lower utility to the risk point removes the
one risk failure but also 14 qualifying views. The two checks are
not redundant, and their intersection trades substantial coverage
for the absence of observed failures in these folds. The stronger
log-risk and lower-utility combination leaves only one source.

These results support history as a utility input as well as a risk
input. They do not support adding the donor signal as a requirement:
history alone has similar MAE, and the release counts differ by one
view. Before a confirmation run, the next useful test is to carry
the history-only design unchanged to a separate development source
with the same horizon and support accounting. Further threshold
search on these 15 sources cannot establish independent transfer.

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_recording_utility_guard.py freeze
.venv-pytorch/bin/python scripts/run_m3w_recording_utility_guard.py score
.venv-pytorch/bin/python scripts/audit_m3w_recording_utility_guard_attribution.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_recording_utility_guard.py \
    tests/test_m3w_recording_history_risk.py tests/test_m3w_recording_risk_pool.py \
    tests/test_m3w_ols_relative_gain.py tests/test_m3w_full_scene_harm_transfer.py \
    tests/test_m3w_dronecrowd*.py tests/test_m3w_recording_forecast_comparison.py
```

Native arm64 local execution; no neural refit or new CREATE job.
Feature extraction took 6.52 s, with resumable per-recording receipts;
the nested score took 1.63 s. The focused suite passed 195 tests.
No confirmation, deployment, manuscript, Stage5C or SMC change.

[Registration](registration.json) · [Frozen utility feature](freeze.json) ·
[Held-source results](analysis.json) · [Guard attribution](attribution.json)
