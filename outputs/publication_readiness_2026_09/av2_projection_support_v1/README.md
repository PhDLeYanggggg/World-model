# AV2: better support prediction does not fix conditional harm

The [selected-path ablation](../av2_selected_projection_risk_v1/README.md)
lost many examples with edited-easy support. I trained a separate support
head on the full donor actor pool and replaced only the support term of
the existing transferred cost. The forecast and gain/harm estimate are
identical in all arms.

## Matched setup

The support head uses observed CV consistency across four history prefixes,
speed and CV/OLS4 disagreement. It does not see the projected forecast or
which action was selected during training. The target is positive observed
CV error no greater than the donor-only easy cutoff. The score for the two
edited actors is one minus the product of their estimated non-support
probabilities. This factorization is an estimator, not a proof of support.

Three arms use the original transferred cost, the same score without its
support factor, or the same score with the new factor. The exact old score
and calibrated held comparator replay. The position projection, two
edited recipients, each step's edit magnitude, native 8/12 grid and the
original dual-floor and 2% rules stay fixed. Four donor cities fit each
head; separate calibration selects the original all-qualified prefix;
the sixth city is held for development scoring. The predictions were
frozen before reading outcomes. Seeds 17, 29 and 43 yield 18 fits and
97,032 donor actor rows counted across folds, not independent actors.

## Support versus harm

There are 340 distinct held development scenarios, 1,020 seed views and
387 views with positive edited-easy support. Of all 1,020, the original
risk scorer identifies 633 unsupported, 213 with observed positive harm,
one failing only the completion bound, and 173 passing risk. Of the
173 risk passes, 119 pass both motion floors. These counts exactly
reproduce the frozen position outcomes.

| Support estimate | Held fold mean AP | City-weighted Brier |
|---|---:|---:|
| Old transferred cost | 0.8322 | 0.1538 |
| All-donor actor head | 0.8767 | 0.1165 |

The paired Brier change is -0.0373 (95% city interval [-0.0493, -0.0269]).
Better support prediction does not establish better harm prediction.

| Rank at fixed five per fold (90 views) | Qualifies | Risk fails | No edited-easy reference |
|---|---:|---:|---:|
| Old cost | 31 | 45 | 10 |
| Remove support | 10 | 71 | 54 |
| Learned support | 31 | 46 | 6 |
| Known support only (hindsight, same remaining score) | 27 | 47 | 0 |

At equal counts, replacing support gives 0.00 pp change in qualification
(paired city 95% CI [-5.56, +5.56]). The oracle only substitutes known
support and leaves gain and harm ranking unchanged; it is not the maximum
possible qualification. Even after excluding every unsupported view, 47
of 90 fail the risk gate. Support is necessary but not the only error
source. Removing it makes the ranking worse.

## Actual release

| Arm | Qualifying calibration prefix views | Held release / qualifies | Held risk fails |
|---|---:|---:|---:|
| Old cost | 6 | 6 / 3 | 3 |
| Remove support | 3 | 3 / 0 | 3 |
| Learned support | 6 | 8 / 5 | 3 |

All three risk failures with learned support are observed positive harm,
not absent easy references. That arm's eight views have known ADE/FDE
0.13742/0.38027 and worst released p95 ADE 1.129 in native coordinates.
On the same eight, conservative gain versus CV/OLS4 is +0.769/+14.735
percentage points. Across the full CV-fallback policy, conservative CV
gain is +0.0091 pp (paired city 95% CI [-0.0039, +0.0241]).
Neither positive mean gain nor better support calibration clears the
original gate. No deployment change.

Preserve support as a distinct component, but the next decisive test
is conditional observed harm **on supported fixed projections**,
using the same predictor, legacy gain and matched counts. Do not fit
another support-only ranker or relax the 2% rule.

## Reproduce

Native arm64 CPU, four Torch threads, zero loader workers. Fitting and
freezing took 33.76 s (700 MB peak RSS); checkpoints every 50 steps
retain the model, optimizer and RNG. Rerun `fit` to resume. The
heartbeat names the process and fold. Calibration reuses prior scalar
receipts; scoring reads held development labels only to annotate support
and risk causes, verifies the old risk flags, and writes immutable
annotation receipts. No CREATE job needed.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_projection_support.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_projection_support.py score
.venv-pytorch/bin/python scripts/plot_m3w_av2_projection_support.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2_*.py tests/test_m3w_cv_anchor_budget.py tests/test_m3w_cv_joint_feasibility.py
```

The figure is generated as local `av2_projection_support.svg` and
`av2_projection_support.png` under the experiment submission directory.
It does not alter the manuscript. [Freeze](freeze.json),
[calibration](calibration.json), [scored metrics](summary.json).
No official AV2 validation/test, DroneCrowd independent confirmation,
submission, Stage5C or SMC change.
