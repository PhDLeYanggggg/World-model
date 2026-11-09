# OpenDD: retraining the scores on the expanded forecast menu

The [expanded menu](../opendd_self_menu_v1/README.md) had more hindsight
headroom, but the old selector had never trained on the new self-history
path. I refit the action costs and tested whether combining them jointly
helps when both selectors can use the same paths and spend the same budget.

## Matched comparison

- A small head predicts conservative gain relative to CV, incremental gain
  relative to OLS4, and three signed risk terms. It sees only the eight
  observed positions and the candidate forecast. Donor targets supervise
  training only. The three training amplitudes are 0.25, 0.5, and 1.0 of
  each candidate's CV-relative deviation.
- I trained 21 heads (seven held locations, three seeds, 400 updates per
  fit). The neural forecasts on each donor come from producers that excluded
  both the donor and the outer location. Training easy cutoffs exclude both;
  the original held evaluation cutoffs exclude the outer location. No held
  score was used to choose the loss, model, or thresholds.
- The retrained arms use one shared head. Additive evaluates each action's
  worst predicted reference gain minus its positive predicted excesses, then
  adds the scores. Joint sums the predicted terms first and then applies the
  same rule. This tests joint use of estimated costs, not a new learned
  pairwise geometry term. Scores are local decision surrogates, not the
  actual recording-level risk test.
- Every arm sees the same expanded menu. Each one fills the original
  3,690 edit slots at their exact peak displacements, with distinct recipients
  and at least one CV recipient per query. The 1,556 singleton decisions
  agree between retrained arms. Scoring applies the original 2% rule and
  missing-step debit to all recording views.

| Arm | Risk passes / 606 | Risk and positive gain vs CV and OLS4 / 606 |
| --- | ---: | ---: |
| Frozen additive | 35 | 30 |
| Frozen joint | 40 | 34 |
| Retrained additive | 38 | 32 |
| Retrained joint | 38 | 32 |

| Paired conservative contrast | Mean (pp) | 95% location bootstrap |
| --- | ---: | ---: |
| Retrained additive minus frozen additive | +0.05454 | [-0.04034, +0.16269] |
| Retrained joint minus frozen joint | +0.06062 | [-0.04936, +0.18268] |
| Retrained joint minus retrained additive | -0.00138 | [-0.00742, +0.00361] |

Joint changes 256 of the 1,067 double-edit decisions. It does not produce
more dual passes than retrained additive, even at individual location totals.
The bootstrap draws six estimable locations 3,000 times after averaging
recording-seed views within each location. A seventh has no edits and no
estimable matched contrast. The 606 views are 202 recordings across three
seeds, not independent samples.

**Decision:** retraining costs on the new menu does not establish that
joint selection helps. The smaller joint-minus-additive difference does
not imply a reliable improvement. The previous exact-slot oracle still
finds a feasible advantage in only 25/439 edited views relative to the
original base additive reference, which is a different comparator. The
next useful question is whether the placement of the available edits across
windows limits the whole policy. Do not scale this same pair decoder. No
reserved outcomes, deployment or manuscript were changed.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/train_m3w_opendd_expanded_costs.py freeze
.venv-pytorch/bin/python scripts/train_m3w_opendd_expanded_costs.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_exact_slot_costs.py tests/test_m3w_opendd_self_menu.py tests/test_m3w_exact_slot_headroom.py tests/test_m3w_slate_shared_menu.py
```

Local arm64 CPU, four Torch threads and no workers. The fit writes atomic
optimizer checkpoints and a PID heartbeat every 50 updates. The same
`freeze` command resumes incomplete fits or reuses completed checkpoints.
Forecast and scoring replay were exact. No source data or weights in Git.
[Registration](registration.json) | [Checkpoints and loss traces](freeze.json) |
[Per-site scores](readout.json).
