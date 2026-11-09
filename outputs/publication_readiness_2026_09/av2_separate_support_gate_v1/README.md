# AV2: a separate support gate does not fix held release

The [conditional harm study](../av2_observed_harm_v1/README.md) found
that ranking by predicted harm can replace observed violations with
cases where the easy reference cannot be measured. I tested whether
requiring support before ranking solves that problem. It does not.

## Fixed comparison

All arms use the same AV2 position forecasts, recipients and two-edit
budget, with eight observed and twelve predicted native steps. The new
arm rejects a view if predicted edited-easy support is below 0.50, then
keeps the original conditional-harm ranking on the remaining views.
Applying the same gate to actor support alone isolates the effect of
conditional harm. A separate 0.75 sensitivity was fixed before scoring;
it was not selected by the held results.

The original 2% rule and both CV/OLS4 accuracy floors stay unchanged.
In each of 18 city/seed folds, the score threshold still comes from the
largest nonempty prefix that satisfies every original check on the
calibration city. Counts and paired city intervals below cover 340 held
development scenarios (1,020 seed views across six cities and seeds
17, 29 and 43), not independent confirmation. Frozen scores and outcome
receipts are reused; the gate replay and comparisons are new. No models
were refit, and no raw held or reserved outcomes were reopened.

## Equal-count comparison

Each fold contributes at most five selections, capped at the fewest
admissible views among its arms. The cap reduces the primary budget from
90 to 89. All numbers in the following table use the same 89.

| Score | Qualifies | Risk fails | Observed harm | No easy reference | Accuracy-floor fails |
|---|---:|---:|---:|---:|---:|
| Transferred cost | 31 | 44 | 34 | 10 | 47 |
| Actor support | 31 | 45 | 39 | 5 | 44 |
| Actor support + 0.50 gate | 31 | 45 | 39 | 5 | 44 |
| Conditional harm | 28 | 44 | 24 | 20 | 45 |
| Conditional harm + 0.50 gate | 28 | 43 | 24 | 19 | 45 |

The 0.50 gate removes one unsupported choice without changing the
qualification or observed-harm count. Compared with the same gate on
actor support, conditional harm reduces observed harm by 16.67 percentage
points (paired city 95% CI [-34.44, -2.22]) but increases missing
reference by 15.56 points ([+6.67, +25.56]). Qualification changes by
-3.33 points ([-13.33, +4.44]). City means weight cities equally; the
intervals are not calculated over overlapping windows or seed views.

At the 0.75 gate, the separate common budget is 86. Against ungated
conditional harm at that budget, missing references fall 19 to 8
(-12.22 points, 95% CI [-21.11, -5.56]) and observed harms rise 23 to 34
(+12.96 points, [+6.30, +19.64]). Risk failures remain 42, and
qualification falls 27 to 26. A support probability is not a harm
probability, even where the reference is likely to exist.

## Actual calibration and held release

| Score | Admitted calibration / held views | Qualifying calibration prefix | Held releases / qualifies | Held risk fails |
|---|---:|---:|---:|---:|
| Transferred cost | 1,020 / 1,020 | 6 | 6 / 3 | 3 |
| Actor support | 1,020 / 1,020 | 6 | 8 / 5 | 3 |
| Actor support + 0.50 gate | 410 / 422 | 6 | 8 / 5 | 3 |
| Conditional harm | 1,020 / 1,020 | 9 | 17 / 4 | 8 |
| Conditional harm + 0.50 gate | 410 / 422 | 9 | 17 / 4 | 8 |
| Conditional harm + 0.75 gate | 237 / 262 | 10 | 11 / 3 | 4 |

The 0.50 gate leaves the exact held release roster unchanged for both
scores. The 0.75 gate removes six releases and adds none. Its remaining
four risk failures include three observed harms and one missing
reference. It still exceeds signed easy harm on two views and fails
at least one accuracy floor on six. Neither variant is deployable.
Next, diagnose which calibration-to-held failures reflect score
misranking among supported views versus genuinely missing support.
Further changes to the support cutoff alone are not supported.

## Reproduce

From the repository root with the frozen parent artifacts and local
scalar receipts, run:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_separate_support_gate.py freeze
.venv-pytorch/bin/python scripts/run_m3w_av2_separate_support_gate.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2_*.py tests/test_m3w_cv_anchor_budget.py tests/test_m3w_cv_joint_feasibility.py
```

The native arm64 CPU replay took 11.07 s to freeze and 11.37 s to score
with peak RSS below 720 MB. The focused suite passed 110 tests. The
[registration](registration.json) and [freeze](freeze.json) were pushed
before scoring. See [calibration](calibration.json) and the
[full results](summary.json) for the city breakdown and intervals.
No new CREATE job, official AV2 validation/test, reserved DroneCrowd
confirmation, manuscript edit, deployment, Stage5C or SMC.
