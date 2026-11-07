# Screening Before Ranking Restores Interventions, Not Risk Control

I fixed the search restriction identified in the matched allocation study.
The same fitted coefficients score the same eight paths, but all actions
are screened against both estimated budgets before selecting up to two per
query. No model is refit, no threshold is tuned, and choices freeze before
this readout. These two deployment-shaped diagnostics use variable counts;
they do not replace the original equal-count component comparison.

| Development result | Joint then budget prefix | Filter before ranking |
|---|---:|---:|
| Actual switches | 12 | 6,184 |
| Nonempty recording views | 9 | 92 |
| Wholly unknown selected rows | 0 | 51 |
| Completion lower vs CV (pp) | -0.00084 [-0.00279, +0.00026] | +0.07746 [+0.02877, +0.13321] |
| Original supported passes / 102 | 3 | 2 |
| Undefined easy risk | 99 | 36 |
| Easy completion risk over 2% | 0 | 64 |

The change fixes the search pathology and recovers a small conservative
utility gain on exposed A, but it does not recover the risk contract. The
6,184 actions are 1.64% of all seed-rows. Known-label utility is +0.08474 pp
[+0.03776, +0.13911]; both intervals bootstrap the 14 sources, not windows.
Already on known easy outcomes, 63 views exceed 2%. Adding wholly unknown
completion radius raises that count by only one, to 64. All completion
risk exceeds 2% in 68 views, and signed easy net in two. Easy positive harm
and signed net are not interchangeable. Passing the estimated constraints
is not evidence of passing the original risk test, even without wholly
unknown labels. This points to selected-action harm/reference estimation,
not insufficient solver activity, as the remaining risk problem.

The policy is not promoted. The next useful test is source-separated
calibration of selected easy harm and selected reference mass, followed by
untouched confirmation. No external outcomes or role boundaries change.

```sh
.venv-pytorch/bin/python scripts/run_m3w_filter_first_actions.py freeze
.venv-pytorch/bin/python scripts/run_m3w_filter_first_actions.py score
.venv-pytorch/bin/python scripts/run_m3w_filter_first_actions.py replay
```

Fixed at `f9c1c929` before the follow-up scoring. Both readouts match exactly.
Full 102-view scalar outcomes include the unmodified 2% tests. No separate
identity fallback certificate, metric/seconds claim, Stage5C or SMC.

[Protocol](protocol.md) | [Metrics](analysis.json) | [Matched study](../readout.md)
