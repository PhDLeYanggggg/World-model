# From Insertion Retrieval to Whole-Recording Repair

These experiments use the same exposed native image-coordinate
cohort: 34 recordings, 14 source groups and three frozen forecast
seeds (102 repeated views). Observation and prediction lengths are
eight and twelve annotation steps. None is an independent test.
The unchanged direct policy has 2 recording passes,
35 undefined easy-reference views and
8 signed easy violations.

## Matched Model Comparisons

The component column shows first eligible-query successes without
thresholding, except for optional withdrawal, which reports the
number of actual withdrawals after the same frozen insertion.
Factorized value measures observed easy reference with nonnegative
local gain; the other probes require non-loss whole-record rescue.
Recording columns use the original full policy with donor-only
calibration. An empty intervention does not establish safety.

### Table 1. Paired development results, control to treatment

| Comparison | Component probe | Recording passes | Signed easy violations |
| --- | ---: | ---: | ---: |
| Factorized value | 3 to 9 | 2 to 2 | 8 to 8 |
| Whole-record target | 1 to 2 | 2 to 2 | 8 to 8 |
| Joint vs additive exchange | 0 to 0 | 2 to 2 | 8 to 8 |
| Optional withdrawal | 0 withdrawals | 2 to 2 | 9 to 9 |
| Observed head boxes | 0 to 2 | 2 to 2 | 8 to 8 |

The box ablation aligns all eight observed head-box centers exactly
to the original trajectories. Its first-query rescue contrast is
+0.95 pp (paired source-bootstrap 95% interval
[+0.00, +2.86]). Both feasible probes come from
1 recording and neither has a donor-selected cutoff.
Box geometry is not an occlusion label.

## What the Action Set Permits

Among the 35 views without easy reference, 18
have no action to withdraw. With observed labels, same-query
one-for-one exchange repairs 5/35; allowing a later
withdrawal at the same recording count raises this to 8/35.
Insertion alone permits 22/35;
an insertion with optional ordered deletions permits
33/35. These are different
action sets and overlapping counts, not results to add or
learned-policy passes. Verified witnesses preserve the original
2% and reference-support checks. More flexible action sets do not
resolve the problem of retrieving the right action from past data.
No deployment, physical-safety or metric claim follows.
