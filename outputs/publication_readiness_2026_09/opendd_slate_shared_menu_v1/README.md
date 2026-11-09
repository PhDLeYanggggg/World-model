# Does the pair score help when both policies can choose the same paths?

In the [earlier slate experiment](../source_excluded_slate_v1/README.md),
the two policies made the same number of edits with the same displacement
budget, but they did not search the same paths. The joint policy picked from
the original forecasts; the matched additive control could also shorten a
forecast to fill a budget slot. I reran the comparison with both policies
searching that same set of shortened or unchanged paths.

## What changed

The inputs, fitted weights, CV and OLS4 reference paths, original intervention
count, and exact peak displacement in each query stay fixed. The menu consists
of OLS4 and the median-slope forecast at each required peak. A candidate can
be shortened, never enlarged. Every assignment must edit distinct agents and
leave at least one CV forecast. Both arms enumerate the same legal assignments;
only the joint arm adds the checkpoint's learned pair score. The one-edit
queries are a null control: the two arms must agree.

I used the 21 existing OpenDD checkpoints, seven held development locations
at three seeds. Each fit excludes its scored location. Predictions were
frozen before decoding label arrays for this run. The original 2% risk rule,
the donor-only easy threshold and the missing-step debit are unchanged.
Neither the WEPDTOF external groups nor the DroneCrowd confirmation groups
entered this run. There is no new training, calibration or threshold search.

## Results

| Measure | Shared-menu additive | Shared-menu joint |
|---|---:|---:|
| Edits | 3,690 | 3,690 |
| Recording-seed views that pass the original risk rule | 39 / 606 | 41 / 606 |
| Views that pass risk and beat both references | 32 / 606 | 34 / 606 |

There were three dual passes only with the pair score and one only without
it, all at the same location. Among the 1,067 double-edit queries, the pair
score changed 693 assignments. The 1,556 single-edit queries agreed, as
required. Compared with the previous result, the shared menu changes 234
joint queries and one additive query.

| Joint minus additive; positive is better | Site mean | 95% location bootstrap |
|---|---:|---:|
| Observed-step gain (pp of CV reference error) | +0.00420 | [-0.02407, +0.03170] |
| Missing-step conservative gain (pp of CV reference error) | -0.01003 | [-0.04651, +0.01356] |
| Observed pair-relative error reduction (native OpenDD units) | +0.000009 | [-0.000209, +0.000189] |

The 606 views are 202 exposed recordings evaluated with three fits. The
bootstrap draws 3,000 samples over six estimable locations after averaging
records and seeds within each location. The seventh location has no edits
and no matched utility estimate. Training folds overlap, so these are
descriptive development intervals, not an independent safety certificate.
The largest share of assignment changes is at `rdb6` (452 of 693); that
location's conservative gain is negative (-0.09615 pp). Pair-relative error
is a geometry diagnostic, not a measure of collision risk.

## Interpretation

The old menu difference was real, but correcting it does not uncover a
reliable benefit. The score is active, not ignored by the solver. It often
changes decisions and still does not reliably reduce error. This points to
the learned pair target rather than another expansion of the search menu.
The current imitation target rewards recovery of a recording-level edit
plan; it does not directly supervise whether a specific pair of edits
improves relative motion beyond the two unary decisions. That is a testable
next hypothesis, not a result of this run.

[Scene Transformer](https://openreview.net/pdf?id=Wm3EA5OlHsG) and
[JFP](https://proceedings.mlr.press/v205/luo23a/luo23a.pdf) study joint motion
prediction. This comparison instead holds the forecast generator and budget
fixed and asks whether a pair score improves the *decision to edit*. It is
not a head-to-head benchmark with those forecasters.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/audit_m3w_slate_shared_menu.py freeze
.venv-pytorch/bin/python scripts/audit_m3w_slate_shared_menu.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_slate_shared_menu.py tests/test_m3w_slate_imitation.py
```

`fresh_run`: shared-menu inference and scoring. `cached_verified`: original
checkpoints, original forecast budgets and exposed OpenDD cohorts. Eight
scoped tests passed. The freeze replay compares all arrays with the prior
run; the score replay requires identical aggregate JSON. The result uses
eight observed and twelve forecast steps in native OpenDD coordinates.
No paper file, deployed policy or registration was changed. Stage5C and SMC
were not run.
