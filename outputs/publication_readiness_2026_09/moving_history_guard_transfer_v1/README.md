# MovingDroneCrowd++: Does the DroneCrowd History Guard Transfer?

On [DroneCrowd](../dronecrowd_recording_utility_guard_v1/README.md),
observed motion improved estimates of gain and harm. I tested the
same history-only calibration on another already exposed development
cohort, without fitting on the target's results.

## Design

I retain all 24 Moving recordings from the previous nine-source
development experiment, covering 69,746 past-eligible rows. The nine
registered source groups are disjoint from the 14 Moving groups that
trained the three CV-target selectors. I use those frozen selector
weights, OLS4 as the default forecast, and the same independent rule
for choosing analytic replacements. The original Moving source-fold
easy and hard cutoffs are retained. This is a full-scene diagnostic;
it does not replace the earlier matched sparse count-and-peak study.

The history features use the first four observations to predict the
remaining four observations, with the frozen candidate identity.
Risk and utility calibration use all 15 DroneCrowd calibration groups
and the same ridge specifications as before. The residual margins
come from holding out whole DroneCrowd sources. Moving contributes no
outcomes to any coefficient, margin or release choice. The choices
and release flags were frozen before scoring. Where the history
risk denominator is undefined, the gate abstains and the recording
stays in the score. The first attempt stopped on that edge case;
the final run records unsupported views rather than dropping them.

The same CV and OLS4 references, missing-label debit, and original
2% all/easy/signed risk checks are used. The 72 views repeat the 24
recordings across three seeds; uncertainty uses 3,000 paired draws
over nine sources. The observation and forecast spans are eight and
twelve native annotation steps. Errors are in image coordinates.

## Results

| Forecast | Available ADE / endpoint FDE (px) | Risk pass | Conservative gain vs OLS4 (pp) |
|---|---:|---:|---:|
| CV | 100.168 / 233.593 | 0/24, undefined edited support | -12.129 |
| OLS4 | 94.669 / 210.317 | 0/24 | 0.000 |
| Fitted-origin OLS4 | 96.065 / 210.811 | 0/24 | -2.072 |
| Frozen independent choice | 94.779 / 209.417 | 0/72 | +0.079 [-1.063, +1.571] |

The independent paths gain +0.837 pp on available labels, with
a 0.757 pp missing-label debit. The remaining +0.079 pp is not a
stable accuracy improvement. Of 69,746 rows, 67,854 have at least
one future label and 45,330 have all twelve; no row is excluded
because of missing future labels. The complete roster's worst
recording p95 available ADE is 608.84 px for the independent
forecast and 619.87 px for OLS4.

None of the three frozen guards selects a recording-seed view.
Historical risk support is defined in 63/72 views. With observed
outcomes, **all 72** exceed 2% all-sample risk; of the 66 with
defined easy risk, **all 66** exceed 2% there too. No independent
forecast could be released by a perfect gate under this rule.
Abstentions are not treated as risk passes.

The history predictor also does not retain its DroneCrowd advantage:
on 63 comparable views, its all-risk MAE is 34.205 pp versus 6.173 pp
for the raw donor signal. Easy-risk MAE is large for both (2673.253
and 2773.444 pp); these are harm-to-reference ratios, not failure
probabilities, and can exceed 100%. Utility MAE is 2.360 versus
2.437 pp. The last two comparisons are descriptive, not evidence of
generalizable lift.

This closes off another calibration-only explanation: there is no
qualifying path among the frozen independent forecasts on this cohort.
I would next test a better forecast bank on an existing development
cohort with fuller labels and a matched edit budget, before tuning
another release gate. Moving is a negative cross-domain development
result, not an independent confirmation or a deployment change.

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_moving_history_guard_transfer.py freeze
.venv-pytorch/bin/python scripts/run_m3w_moving_history_guard_transfer.py score
.venv-pytorch/bin/python scripts/run_m3w_moving_history_guard_transfer.py replay
.venv-pytorch/bin/python scripts/audit_m3w_moving_history_guard_transfer.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_moving_history_guard_transfer.py \
    tests/test_m3w_recording_utility_guard.py tests/test_m3w_recording_history_risk.py \
    tests/test_m3w_recording_risk_pool.py tests/test_m3w_ols_relative_gain.py \
    tests/test_m3w_full_scene_harm_transfer.py tests/test_m3w_dronecrowd*.py \
    tests/test_m3w_recording_forecast_comparison.py
```

Native arm64 with four Torch compute threads and no data workers.
Local freeze took 4.37 s, and scoring 4.99 s (0.55 GB maximum RSS).
Both are resumable by recording. Replay and attribution read cached
scores only. The focused suite passed 196 tests. No new neural fit,
CREATE job, Moving A outcome read, DroneCrowd confirmation read,
manuscript edit, submission, Stage5C or SMC.

[Registration](registration.json) · [Frozen choices](freeze.json) ·
[Results](analysis.json) · [Failure attribution](attribution.json)
