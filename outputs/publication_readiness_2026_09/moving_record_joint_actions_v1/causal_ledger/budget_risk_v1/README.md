# Pedestrian Candidate Bank and Query Budgets

[Sparse support](../sparse_support_v1/README.md) did not close the
OLS4 gap. This experiment separates the forecast bank from the edit
budget on the same 14 Moving A source groups, 34 recordings, three
forecast seeds, and native 8/12 protocol (102 repeated views).

## Exact Action Capacity

The original bank and an expanded bank use the same cohort and
reference. Seven past-only additions are OLS3/5/6, Theil--Sen8,
two fixed velocity blends, and bounded acceleration. New paths
are clipped to each row's original maximum displacement from CV.
For the exact diagnostic, select only known, CV-improving actions,
at most one per row. Require at least one easy edit and limit how
many agents can be edited per query. With no recording cap the
top distinct rows in each query maximize gain. If there is no
easy edit, the least costly easy replacement is the exact forced
anchor. Every successful witness is rescored.

| Cap per query | Original bank, all / strong | Expanded bank, all / strong |
| --- | ---: | ---: |
| 1 | 48 / 0 | 48 / 0 |
| 2 | 48 / 0 | 48 / 0 |
| 4 | 59 / 8 | 60 / 9 |
| 8 | 72 / 15 | 72 / 15 |
| No cap | 93 / 36 | 93 / 36 |

The first column in each bank counts joint passes out of 102;
the second counts joint passes out of 36 views where OLS4 is
certified better than CV. More actions matter more than these
seven variants. Uncapped witnesses can change many rows
(326,565 edits across original-bank successful views). These
are outcome-aware limits in a restricted, known-improving class,
not general 2% upper bounds or online policies.

## Causal, Query-Count-Matched Test

On the original bank, train a head on the seven frozen fitting
sources of each fold to estimate conservative CV gain and all/easy
positive-harm excess. Its screen uses only observed-history
features at inference. It admits positive predicted gain and
nonpositive predicted excess, then takes up to the cap at each
query. Gain-only and risk-blind learned controls receive exactly
that query's realized count. The six reserved sources are not
used to set a threshold. Estimates do not certify the risk rule.

| Cap | Matched edits | Screen / old gain / blind joint pass | Screen risk pass |
| --- | ---: | ---: | ---: |
| 1 | 3,499 | 1 / 1 / 0 | 3 |
| 4 | 12,235 | 1 / 1 / 0 | 2 |
| 8 | 21,372 | 1 / 1 / 0 | 2 |

Cap 4 screen versus old gain is -2.899 pp [-4.960, -1.211]
conservative utility. It includes -1.034 pp observed difference
and 1.865 pp paired unknown-disagreement debit. Compared with
the screen at cap 1, cap 4 improves observed utility by
+0.237 pp [+0.009, +0.540], but its debit is 0.651 pp and the
conservative contrast is -0.414 pp [-0.963, -0.040].
Intervals resample the 14 source groups 3,000 times after
averaging seeds. The debit is a conservative component on
unobserved outcomes, not measured harm. No screened arm jointly
passes on any of the 36 strong-OLS4 views.

The capacity gain is real within the defined class, but the
learned screen does not identify it. A specific follow-up is
to estimate wholly unobserved-label probability from causal
features, then apply the known path disagreement explicitly.
Compare against the entangled cost target at the same query
counts. Future masks must remain labels only.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/audit_m3w_moving_capacity_factorial.py
.venv-pytorch/bin/python scripts/run_m3w_moving_budget_risk.py
.venv-pytorch/bin/python scripts/run_m3w_moving_budget_risk.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_moving_budget_risk.py
.venv-pytorch/bin/python scripts/build_m3w_moving_budget_replay.py
```

`fresh_run`: exact budget/forecast factorial and 42 model fits,
21,000 updates, four arm64 Torch threads, zero loader workers,
PID 83546, 39.0 seconds. Checkpoint and heartbeat every 100
steps. `cached_verified`: existing forecast lineage, source
roles, zero-update replay, and frozen debit attribution.
`not_run`: new CREATE training or independent confirmation.
Original 2% and undefined-risk handling remain; no deployment,
submission, metric/seconds claim, Stage5C or SMC.
