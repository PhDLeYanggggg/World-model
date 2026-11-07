# Acquiring Easy Reference And Repairing The Rest Of The Recording

The original direct stream leaves 35 of 102 exposed Moving A views
without a defined easy upper bound. In 18 it never switches; in
17 it switches but selects no easy-reference action. Deletion
alone cannot add reference mass in either group. This diagnostic
keeps the original 2% risk and reference-support rules, adds one
bank action to a previously unswitched row, and requires
nonnegative paired completion-lower utility versus direct.
The task uses eight observed and twelve future annotation steps
in native image coordinates.

## Capacity, Not Model Accuracy

| Allowed edit | No original switch / 18 | Active without easy reference / 17 | Checked feasible / 35 |
|---|---:|---:|---:|
| Insert one, no deletion | 18 | 4 | 22 |
| Insert before the first direct switch, then allow deletion | 18 | 11 | 29 |
| Insert anywhere, delete no earlier than insertion | 18 | 15 | 33 |
| No ordering restriction | 18 | 15 | 33 |
| Ordered edit, insertion restricted to old eight-slot lists | 12 | 6 | 18 |

All feasible cases are solver-optimal for minimum deletion count
and checked by the unchanged original scorer. There are no
unresolved cases. With the full bank and ordered edits, two
views are infeasible. Under the eight-slot list, 12 have no eligible
insertion and five more are infeasible. With ordered full-bank
edits, 30 of 33 returned witnesses have strictly positive paired
lower gain; the other three are nonnegative. The solver minimizes
deletions, not gain, so this is a count of positive witnesses, not
a ceiling on the existence of a positive-gain witness.

The ordered joint edit rescues 11 more views than insertion alone.
It also rescues four more than insisting on an opening before
direct begins. Allowing deletion before support arrives does not
add a feasible view. The 33 checked witnesses delete 24 original
actions in total, with insertion and deletion in the same query
in two views. The original per-query cap remains in force.

This isolates the important design choice: the old shortlist
loses 15 of the 33 achievable repairs even if the downstream
decision sees the labels. Keeping forecast families is not enough
if the shortlist excludes the row that supplies easy reference.
A support-specific retrieval path, followed by jointly scoring
the anchor and removable actions, is a more directly motivated
next experiment than relaxing chronology or refitting the same
deletion threshold. The known easy label is not an inference
feature; the choice would have to be predicted from past data.

## Exact Accounting

For a direct action removed on a wholly unknown row, the paired
lower gain pays the full path disagreement; deletion does not
turn missingness into a benefit. Easy insertion candidates have
the same all and easy excess, and the signed easy cost is the
negative observed gain. For one insertion only, dominance
within a query is therefore exact when an alternative has no
greater excess and no smaller gain. We recheck every returned
witness for risk, non-loss and the query cap rather than taking
the solver status as an observed pass.

## Reproduction

Registered in `6778aa11` before the solver run. The direct bank
and shortlist heads are verified parents; no new model is fitted.
The fresh solve checkpoints by recording and seed. Aggregate
replay reads those receipts rather than re-solving. With the
private parents present, run from the repository root:

```sh
.venv-pytorch/bin/python scripts/run_m3w_support_acquisition.py score
.venv-pytorch/bin/python scripts/run_m3w_support_acquisition.py replay
.venv-pytorch/bin/python scripts/plot_m3w_support_acquisition.py
```

The figure is regenerated from the public scalar analysis and
matching score/replay receipts; it does not need raw data. The
35 views are a retrospective subset of 34 recordings repeated
across three forecaster seeds, not an inference-time eligibility
rule. No oracle decisions are exported. No model pass, independent
evaluation, deployment, metric/seconds, physical safety, Stage5C
or SMC claim follows.
