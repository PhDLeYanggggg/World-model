# Label-Aware Whole-Query Capacity Diagnostic

This diagnostic asks what the already frozen joint proposal slate
permits if the chooser knows outcomes. It is not a learned policy,
additional calibration search or change to the 2% rule.

For each recording-seed view, solve two slates. The first has the
existing ungated joint proposal or abstention per query. The second
also permits the existing independent, shared-family and pair-additive
options, but never combines parts of different proposals. Counts can
vary by query and recording. This is a capacity bound, not a matched
policy comparator.

Use the original positive-harm all/easy risk excesses, the original
signed easy net cap and positive observed easy-reference support.
Charge wholly unknown changed rows their original maximum trajectory
disagreement. Maximize the conservative gain versus CV and versus
OLS4 separately, using query-weighted mass and the same CV denominator.
The latter objective retains the CV-to-OLS4 uncertainty on unchanged
rows, and updates it only where a forecast changes. Use one second per
MILP. Distinguish verified witnesses, solver infeasibility, unresolved
time limits and nonpositive objective upper bounds. Recheck every
reported witness with the original recording scorer.

Run first on the 102 exposed development views using the registered
leave-one-source-out controller. A separate calibration-only diagnostic
may run after the original 40-recording result and its replay, without
updating or selecting a policy. Never read the confirmation role. Report
counts and median witness utility only; do not export any hindsight
decision, reference-derived features or new model. Units are pixels
and annotation steps, not calibrated distance or seconds.
