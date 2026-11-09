# Separate recipient nomination from path replacement

Use the same 24 exposed recordings, nine source groups, and three
three-way source rotations as v135. Keep the original CV-anchored
2% and support rule, OLS4 comparison, and native 8/12 protocol.
No independent confirmation or additional A outcomes enter this run.

Construct train-only recipient frequencies from multiple verified
recording plans. Start from the two existing analytic witnesses,
try deterministic and random deletions while retaining the original
constraints, and independently rescore every unique plan before use.
The frequencies describe this witness family, not all feasible plans.
Compare a single-plan target and the ensemble target using identical
features, architecture, sampling and update budgets.

Separately train a within-recipient path preference model on dual
utility and original risk targets, including an action-pair loss.
Inference uses causal motion, baseline differences and query context.
For the held ablation, reuse v135's cost-only per-query edit counts.
Cross cost/single/ensemble recipient ranks with the frozen cost or new
path rule. The new path must be no farther from CV than the frozen
cost model's best legal action for that same recipient. Report
geometry, conservative and observed accuracy, easy risk and coverage.
Keep the earlier frozen policy as an additional unequal-count reference.

For prospective use, choose among the two ensemble-path options on
calibration only, with query fractions 10/25/50% and predicted score
margins 0/.05. To qualify, the original risk/support rule must pass
on every calibration recording and conservative utility must be
positive versus both CV and OLS4. Otherwise return CV.
