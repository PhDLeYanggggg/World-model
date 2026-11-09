# Minimum-edit supervision for full-bank intervention

Use the same 24 exposed development recordings, three seeds and
three source-role rotations. First bound minimum recording-level edit
counts in the full bank under the original CV risk/support checks and
positive conservative gains over CV and OLS4. Keep independently
verified full-bank incumbents; report lower bounds and verified upper
bounds separately at 5/10/25/50/75% edit budgets. Existing negative
full-bank bounds are not changed. This is a hindsight diagnostic.

For each training role, minimize edit counts on the analytic path
bank, starting from previously verified training-only witnesses.
Each retained target is independently rescored. Outcome information
is used for supervision only. Compare cost-only and teacher-augmented
scorers with the same architecture, supervised risk and dual utility
targets, and shared sampling stream. The teacher arm additionally
learns which path actions enter minimum-edit plans. Both use causal
observed motion, baseline disagreement and query context. No future
labels, known ease, realized risk, or teacher actions enter inference.

Calibrate only on the matching calibration role. Test equal caps
and exact per-query matched counts on the held development role.
An original-rule qualifying setting must beat both references and
pass every calibration recording; otherwise retain CV. No A,
DroneCrowd calibration or independent confirmation outcomes are opened.
