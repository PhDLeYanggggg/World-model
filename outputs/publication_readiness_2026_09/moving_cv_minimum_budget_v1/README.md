# How many edits are needed, and can a model learn which ones?

The previous full-bank witnesses used a median 89.24% of rows.
That was the edit rate of one set of feasible solutions, not the
edit rate required. I kept the forecasts and original CV-anchored
2% rule fixed, computed lower bounds using the cardinality dual,
and independently rescored reduced sets of edits for upper bounds.

| Maximum fraction of a recording | Verified | Excluded | No support | Unresolved |
|---|---:|---:|---:|---:|
| 5% | 12 | 51 | 6 | 3 |
| 10% | 18 | 42 | 6 | 6 |
| 25% | 38 | 27 | 6 | 1 |
| 50% | 49 | 15 | 6 | 2 |
| 75% | 52 | 13 | 6 | 1 |

The median minimum among the 53 feasible views is bounded by
12.01% and 14.61%; three individual minima were exactly bounded.
These are recording budgets on the exposed development set.
They allow hindsight reallocation, unlike a causal per-query cap.
The generic integer search on held recordings was stopped after
the faster certificate and deletion method covered all 72 views.
Its partial checkpoints are retained but are not counted here.

## Fresh training

Within the three training roles, integer search reduced 17
verified analytic plans from 54,465 to 49,161 edits. Each plan was
rescored. The cost-only and auxiliary imitation models saw the
same inputs and sampling, including oversampled teacher-selected
actions. The teacher arm also learned which actions were selected.
Inputs were causal motion, query context and baseline disagreement;
outcomes were used only in targets and scoring. There were 18
native PyTorch fits, 21,600 updates, three seeds and three role folds.

| Held development arm | Edits | Original risk passes / 72 | Worst known signed easy degradation |
|---|---:|---:|---:|
| Frozen comparison | 54,748 | 14 | 61.90% |
| Cost-only | 45,698 | 0 | 291.50% |
| Teacher at equal query cap | 39,180 | 0 | 399.93% |
| Teacher at exact cost-only counts | 45,698 | 0 | 399.93% |
| Teacher with predicted prefix guard | 3,147 | 3 | 119.22% |

At exact per-query counts, imitation minus cost-only is **-14.054 pp**
on the conservative full-grid comparison (95% source-bootstrap
interval -24.242 to -5.438). The observed-step contrast is -3.405 pp
(-10.191 to +0.298). The conservative pair-distance contrast is
-15.122 pixels (-29.871 to -4.337). The risk guard sharply reduced
coverage without qualifying a strategy. Calibration: **0/9**.
Deployment stays unchanged.

## Why it failed

Changing recipients accounts for 9.076 of the 10.650 pp missing-step
disagreement debit (85.2%). On the same recipients, selecting the
teacher's preferred path changes the conservative contrast to
-5.136 pp (-9.351 to -1.617). Clipping each replacement to no more
than the cost-only path's peak displacement narrows the deficit to
-2.704 pp (-5.341 to -0.609). Its observed-step difference is near
zero, and original risk passes remain zero. These are frozen
diagnostics, not model selection. The next experiment should rank
recipients before selecting displacement-limited replacements,
using supervision that tolerates multiple feasible plans rather
than treating one hindsight solution as the unique correct action.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_cv_minimum_budget.py teacher --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_budget_certificates.py
.venv-pytorch/bin/python scripts/train_m3w_cv_minimum_budget.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_minimum_budget_failures.py
.venv-pytorch/bin/python scripts/build_m3w_cv_minimum_budget_replay.py
```

Training, certificates and the failure diagnosis are `fresh_run`.
Frozen forecasts and exact no-update replay are `cached_verified`.
The anonymous NumPy replay covers the scalar policy and budget
tables, not training, pair geometry or raw observations.
This work uses native 8 observed / 12 future steps and dataset-local
pixels. No metric or physical-safety claim. DroneCrowd's 30 independent
confirmation groups remain sealed; Stage5C and SMC were not run.

[Protocol](protocol.md) | [Budget table](capacity_certificates.json) |
[Training](training_readout.json) | [Failure diagnosis](failure_diagnosis.json) |
[Manuscript v135](../evidence_manuscript_v135/README.md)
