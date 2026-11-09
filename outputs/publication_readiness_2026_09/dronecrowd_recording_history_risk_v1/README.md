# DroneCrowd: Does Observed Motion Predict Recording-Level Harm?

The [previous recording-level calibration](../dronecrowd_recording_risk_pool_v1/README.md)
missed 21 of 25 views that had a qualifying forecast. This experiment
holds the four forecast arms, paths, gain scores and replacement counts
fixed, and compares two sources of evidence about harm.

## Design

For each eight-position input I fit the same analytic path family on
the first four positions and score it against the last four observed
positions. I use the candidate identity and changed-from-CV mask of
the corresponding frozen twelve-step forecast. The historical easy
proxy uses the old easy cutoff multiplied by 5/13, the ratio of the
mean index of the two forecast spans. This is a history feature, not
an independent risk observation; the final choices also depend on the
eight inputs. All actual risk scoring still uses the original cutoff,
both 2% limits, label support and missing-label accounting.

The unit is a complete recording, not a rolling release. I aggregate
the history feature and the donor model's predicted risk on the same
four forecast arms. For each arm and risk endpoint I fit the same
fixed-penalty ridge to `log1p(100 * risk)` on the other fourteen
source groups, with equal weight per source. I compare donor-only,
history-only and combined inputs. A point rule uses the predicted
risk directly; a nested residual envelope adds the largest positive
inner held-source underprediction. For each outer source, neither
its calibration predictions nor the envelope use its labels. The
point rule chooses the best predicted positive OLS4 gain among arms
below both limits. The history feature was frozen before fitting.

I use the 40 previously opened calibration recordings across 15
source groups, with three frozen donor seeds (120 views). Risk MAE
intervals use 3,000 paired source bootstrap draws. No trajectory or
future label was newly read in calibration: it reuses the exact
recording score summaries. The 30 independent sources remain unused.
Coordinates are image-space; evaluation uses eight observed and
twelve forecast annotation steps.

## Results

| Risk signal and calibration | All/easy risk MAE (pp) | Point rule selected | Actual risk pass | Both accuracy floors and risk |
|---|---:|---:|---:|---:|
| Previous donor offset | 1.640 / 2.076 | 13 | 12 | 4 |
| Donor ridge | 2.025 / 1.822 | 1 | 1 | 0 |
| Observed history ridge | 0.808 / 0.511 | 36 | 25 | 25 |
| Combined ridge | 0.877 / 0.520 | 34 | 24 | 24 |

History alone reduces mean risk MAE by **1.264 pp [0.845, 1.723]**
relative to the matched donor ridge; adding history to the donor
signal reduces it by **1.225 pp [0.836, 1.653]**. The history point
rule recovers all 25 opportunities but also selects 11 views that
fail actual risk. The absolute residual envelope abstains everywhere.

In a follow-up on the same development partition, I kept the point
predictions fixed and computed the nested residual envelope on the
ridge's log-risk scale instead. The history-only rule selects 18 views
from **two** sources: all 18 meet actual risk, and 11 also beat both
accuracy floors. The combined rule selects 18 from the same two
sources: 18 meet risk and 10 also beat both accuracy floors. Donor
alone selects none. This is a narrow development result, not a new
risk guarantee or evidence that a frozen rule transfers to other
sources.

I also replaced each selected history-only output by the independent
arm on exactly the same recording-seed views and query counts. The
independent paths pass both accuracy floors and risk in all **18**
views, compared with 11 for the risk-priced menu. The menu loses
0.415 and 0.720 pp against independent selection in the two sources.
However, when restricted to the independent arm **before selection**,
the same log envelope selects just seven views; all seven pass both
floors and risk. The matched outcome comparison is not evidence that
the independent gate could have released all 18 prospectively.

This supports observed history as a better risk input. It does not
support using risk prices to replace otherwise useful forecasts. The
next comparison should keep independent forecasts as the primary arm
and test a predeclared history-based risk guard with a distinct utility
check, rather than fit new penalties on these two groups. Do not use
this development follow-up as independent confirmation.

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_recording_history_risk.py freeze
.venv-pytorch/bin/python scripts/run_m3w_recording_history_risk.py score
.venv-pytorch/bin/python scripts/audit_m3w_history_risk_relative_envelope.py
.venv-pytorch/bin/python scripts/audit_m3w_history_risk_release_control.py
.venv-pytorch/bin/python scripts/audit_m3w_history_risk_independent_gate.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_recording_history_risk.py \
    tests/test_m3w_recording_risk_pool.py tests/test_m3w_ols_relative_gain.py \
    tests/test_m3w_full_scene_harm_transfer.py tests/test_m3w_dronecrowd*.py \
    tests/test_m3w_recording_forecast_comparison.py
```

Native arm64 local environment; no new neural fit or CREATE job.
Feature extraction took 8.11 s and nested scoring took 1.68 s.
Recording receipts allow restart of feature extraction. The focused
suite passed 193 tests. No deployment, manuscript or submission change;
Stage5C and SMC remain disabled.

[Registration](registration.json) · [Frozen history](freeze.json) ·
[Matched calibration](analysis.json) · [Log envelope](relative_envelope.json) ·
[Matched release control](released_control.json) · [Independent gate](independent_gate.json)
