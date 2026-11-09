# AV2: source-excluded risk training does not improve position release

The [fixed position release study](../av2_position_release_v1/README.md)
showed that a stronger position reference was not enough: the best prior
ranking still failed the original risk conditions on held development
cities. I trained a score specifically on projected forecasts while
leaving the evaluated position paths, recipients and radii unchanged.

## Source-excluded training

For each rotation, one city is held, a separate city calibrates, and four
cities supply training examples. A donor city's projection model is fit on
the other three donors. It uses the same heading-aligned position network
as the evaluated projection, with frozen causal anchor profiles as radius
inputs. On the excluded donor city, the nested EqMotion and action models
supply the two profiles. Both also exclude that city, calibration and held
sources. The direction model, action checkpoint and forecast file are
recorded for every donor role. The donor's own future is used only as the
score's supervision, not by the producer or its easy cut.

The confidence MLP learns two conservative gains relative to CV and
OLS4, edited-easy support, and the original risk pass conditional on
support. The **gain-only** arm uses the same two outputs as **gain-risk**,
which additionally multiplies by the support and risk probabilities.
The baselines are unchanged transferred-cost and risk-replay scores.
No comparator is refit. The old confidence scores and projection
outcomes replay exactly. All score forecasts are frozen before scoring;
the largest nonempty all-qualified calibration prefix remains unchanged.

Three seeds (17, 29, 43) across six rotations produce 54 distinct
three-donor direction fits used in 72 donor roles, 18 score fits,
and 265,296 ordered donor options summed across folds. Options and seeds
are not independent. The held readout is 340 development scenarios
(1,020 seed views; 119 forced passes), not independent confirmation.

## Same-count results

The following ranks use a fixed 1, 5 or 10 views per fold, with equal
intervention counts across arms. Ties at the count boundary receive
fractional weight independently of the outcome. These are ranking
diagnostics, not release policies.

| Score | Held top 1 / 18 qualifies | Held top 5 / 90 qualifies | Held top 10 / 180 qualifies | Held fold mean AP | Risk fails at top 5 |
|---|---:|---:|---:|---:|---:|
| Transferred cost | 6 | 31 | 56 | 0.443 | 45 |
| Risk replay | 3 | 24 | 50 | 0.347 | 56 |
| Gain only | 1 | 15 | 27 | 0.202 | 66 |
| Gain and risk | 2 | 15 | 27 | 0.224 | 65 |

At the matched top-five count, the gain-risk score trails transferred
cost by 17.78 percentage points in qualification (paired city 95% CI
[-34.44, 0.00]). Relative to the gain-only ablation, its qualification
difference is 0.00 pp ([-11.11, +12.22]) and its risk-failure
difference is -1.11 pp ([-11.11, +7.78]). Adding risk information
does not demonstrate a same-count benefit in this design.

## Actual release

| Score | Qualifying calibration prefix views | Held released / qualifies | Held risk fails | Held signed easy >2% |
|---|---:|---:|---:|---:|
| Transferred cost | 6 | 6 / 3 | 3 | 1 |
| Risk replay | 9 | 26 / 6 | 18 | 11 |
| Gain only | 1 | 0 / 0 | Not estimable | Not estimable |
| Gain and risk | 5 | 5 / 0 | 4 | 0 |

Three of the five gain-risk releases fail at least one of the motion-floor
checks; this can overlap with risk failure. Its released forecasts have
known ADE/FDE 0.08961/0.21067 and worst released p95 ADE 0.576 in native
units. The conservative CV/OLS4 gains on those same five are
-3.950/+20.126 pp. Across the entire held policy with CV fallback the
conservative CV gain is -0.0052 pp (paired city 95% CI [-0.0271, +0.0088]).
Zero signed easy exceedances in this subset are not a risk pass.

This test rules out the claim that training on any out-of-source projection
is enough to fix the score. Two differences remain unresolved: the score
sees all ordered donor options while deployment scores one selected
projection, and the three-donor position producer is not identical to the
frozen four-donor producer. The next test should isolate those effects on
donor forecasts, particularly whether training on causally selected paths
recovers the upper confidence tail. No threshold change, angle retraining
or release of reserved confirmation data is justified by this result.

## Reproduction

Fitting and freezing took 49.80 s and 843 MB peak RSS on native arm64,
four Torch threads and no loader workers. Direction models use 250 updates;
score models use 400. Atomic checkpoints every 50 updates retain weights,
optimizer and RNG; rerunning `fit` resumes. The heartbeat logs PID, fold,
and source. Scoring and the matched-count audit reuse verified scalar
calibration and held receipts, without reading the raw future rows again.
No CREATE job was needed.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_projection_risk.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_projection_risk.py score
.venv-pytorch/bin/python scripts/audit_m3w_av2_projection_risk_matched.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2_*.py tests/test_m3w_cv_anchor_budget.py tests/test_m3w_cv_joint_feasibility.py
```

[Freeze](freeze.json), [calibration](calibration.json), [metrics](summary.json),
[paired same-count audit](matched_audit.json). No official AV2
validation/test, DroneCrowd independent confirmation, manuscript,
submission, deployment, Stage5C or SMC change.
