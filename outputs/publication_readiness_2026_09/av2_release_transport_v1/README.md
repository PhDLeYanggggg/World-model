# AV2: matching calibration coverage does not fix the ranking

The [support gate test](../av2_separate_support_gate_v1/README.md)
left actual releases essentially unchanged. I checked whether that
failure came from a score threshold transferring poorly between cities
or from the ordering of the forecasts themselves. Both occur; a
threshold or coverage adjustment alone cannot recover most of the
available forecasts.

## Method

The unchanged bank supplies one fixed two-edit position forecast per
scenario and seed. The protocol uses eight observed and twelve predicted
native steps, the original 2% risk checks and both CV/OLS4 accuracy floors.
The 18 rotations are six development cities by three seeds, with four
donor cities, one calibration city and a distinct held city per rotation.
There are 340 distinct held scenarios and 1,020 seed views. The original
calibration rules and their actual release counts replay for all six
scores before any counterfactual is reported.

1. **Actual policy:** apply the score threshold from calibration.
2. **Count match, diagnostic:** take the same number of top-ranked views
   in the corresponding held fold as passed calibration, capped by
   admitted views. This removes absolute score-scale and coverage changes,
   without changing the ordering. Boundary ties use fractional weights.
3. **Label-aware ceilings:** compute the longest prefix that passes
   every original check in each held fold. A failing score tie blocks
   its entire group. Repeat with the actual presence of an easy
   reference known. The unconstrained count oracle can pick any
   qualifying admitted view at the same fold budget.

The last two analyses cannot replace the deployed threshold. Held
outcomes are used only for diagnosis. Nothing was fitted, reselected or
deployed, and no raw held futures or reserved outcomes were reopened.

## Results

| Frozen score | Calibration count | Actual held released / qualifies | Count match qualifies / budget | Count oracle | Longest all-pass held prefix | With known reference |
|---|---:|---:|---:|---:|---:|---:|
| Transferred cost | 6 | 6 / 3 | 1 / 6 | 6 | 8 | 11 |
| Actor support | 6 | 8 / 5 | 5 / 6 | 6 | 14 | 14 |
| Actor support + 0.50 gate | 6 | 8 / 5 | 5 / 6 | 6 | 14 | 14 |
| Conditional harm | 9 | 17 / 4 | 3 / 9 | 9 | 7 | 10 |
| Conditional harm + 0.50 gate | 9 | 17 / 4 | 3 / 9 | 9 | 8 | 10 |
| Conditional harm + 0.75 gate | 10 | 11 / 3 | 5 / 10 | 10 | 8 | 8 |

Budgets are matched between calibration and held **within an arm**;
the 6-, 9- and 10-view rows are not an equal-budget comparison of scores.
The [previous same-count study](../av2_separate_support_gate_v1/README.md)
provides that comparison. Six folds had a qualifying calibration prefix
for conditional harm. Matching their held count leaves three passes,
three risk failures and six accuracy-floor failures; these failures
overlap. The count oracle could use all nine slots. The city-balanced
shortfall is 0.517 qualifying views per 100 held views (95% city bootstrap
CI [0.139, 0.933]), with 3,000 resamples of the six cities after averaging
seeds. It is a development diagnostic, not an estimate on new cities.

There are **119** qualifying fixed forecasts in hindsight. For
ungated conditional harm, 4 actually qualify on release, 7 additional
qualifying views are inside a held all-pass prefix with known support,
and **108** are below its first failure. The 0.50 gate excludes 13
qualifying views without releasing any additional one. The 0.75 gate
excludes 39. Even perfect reference observability does not repair the
ranking.

The first blocking group with reference known contains 18 views for
conditional harm, one per fold. Seven fail only the CV/OLS4 floor,
three fail only risk and eight fail both. A risk-only prefix could
retain 28 views rather than 10; the extra 18 are not valid releases.
This separates an accuracy-ranking problem from missing references,
without claiming that observed harm is solved. Actor support ranks
better under this diagnostic but still has actual risk failures.

## Decision and reproduction

Do not spend another run tuning the support threshold. A further AV2
method change should address the joint risk-and-accuracy ordering and
compare it at the **same per-fold counts** with the frozen actor-support
score. The earlier all-option gain regressor already failed, so a new
loss must test a different mechanism rather than repeat that fit.
This audit does not change the manuscript or resolve the independent
external evidence gap.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/audit_m3w_av2_release_transport.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2_*.py tests/test_m3w_cv_anchor_budget.py tests/test_m3w_cv_joint_feasibility.py
```

[Full audit and receipt hashes](transport_audit.json). The arm64 CPU
run took 11.50 s at under 710 MB peak RSS; 113 focused tests passed.
No new CREATE job, AV2 official validation/test, DroneCrowd independent
confirmation, submission, Stage5C or SMC.
