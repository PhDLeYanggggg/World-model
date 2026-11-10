# Why WEPDTOF fell short at the original edit budget

The frozen external comparison had 0/45 views passing both the original
risk checks and both reference comparisons. I checked whether a better
recipient ranker could close the OLS4 gap without increasing the number
or radius of edits. This is a post-hoc bound on those same forecasts
and score aggregates, not another external test.

| Paths allowed, risk constraints relaxed | Excluded on observed steps | Excluded under the conservative utility | Median conservative upper bound |
| --- | ---: | ---: | ---: |
| Arbitrary paths within the registered query radii | 24 / 36 | 30 / 36 | -17.18 pp |
| The original two candidate paths per eligible recipient | 25 / 36 | 30 / 36 | -18.83 pp |

The 36 edited views come from 12 clips and 11 source groups. The arbitrary
path conservative bound excludes every edited seed view in 10 clips and
9 source groups. The nine remaining roster views made no edits; they
cannot strictly improve on CV. Among the four views where the frozen
joint arm met the risk checks, all four are excluded even on observed
steps. Those four views are from four distinct source groups. Six edited
views are not excluded by the conservative bound; that is not proof that
any alternative can pass. None of these calculations changes 0/45.

## Calculation

For each known step, changing the forecast by distance `d` can improve
the OLS4 comparison by at most `d`, by the reverse triangle inequality.
On an unknown step the comparison debits the distance to OLS4, which
has the same Lipschitz bound. With `n` online actors, `k` eligible edits
and radius `r` in a query, the increase from CV is at most `12*k*r/n`.
Sum over queries and use the original observed CV error mass to express
percentage points. For the finite menu, replace `12*r` by the exact
sum of waypoint distances for each option and optimize the top `k`.
Multiple frozen reference outputs give valid independent anchors, but
did not tighten the reported exclusion counts over the CV anchor.

I rebuilt the legal menus from the frozen CV, OLS4 and spline outputs,
verified the query budgets and archived selections for all four matched
arms, and checked that every realized score is below its envelope.
The result uses no raw targets, future indices or training; all three 2%
risk checks, the original evaluation and the CV fallback stay unchanged.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_wepdtof_frozen_capacity.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_wepdtof_frozen_capacity.py tests/test_m3w_wepdtof_scalar_replay.py tests/test_m3w_moving_peak_certificate.py
```

Fifteen tests passed, including exhaustive menu checks on partially
observed synthetic examples. [Aggregate receipt](receipt.json).

**Implication.** For most clips here, improving the ranker alone cannot
recover the OLS4 gap at the tested budget. A better starting forecast
should be evaluated before spending more on a joint ranker. Because the
same conclusion holds on observed steps in 24 views, missing labels are
not the only explanation.
