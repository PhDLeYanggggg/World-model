# Fixed Eight-Arm Scaling Readout

This implements the training protocol's unchanged advancement rule before new
candidate development predictions. Admit all 144 final payloads and 72 exact
unit-control training replays. Add the 144 cached-verified quadratic/utility
heads, preserving their original hashes and predictions, and four forest controls.
No model is retrained in readout. All 72 views are required, without label-based
inference filtering. Validate input/target/row/split alignment against the original
TRAIN source and previous development record.

Primary candidate is budget_scaled. Primary comparators: unit_control, quadratic,
task_utility, original forest. Eight full/same-count paired lower utility contrasts
must all have positive nominal 95% locality-bootstrap lower endpoints. Candidate
full and all four primary matched policies must have complete finite completion
support and no positive-easy-harm/reference risk above 2%. Empty or unsupported
policies fail advancement; unknown outcomes remain in conservative bounds.

All other forest comparisons and signed-score MSE are retained, not substituted
for failed primary comparisons. Report risk, selected counts, all original-unit
TRAIN monitors, dual caps, soft/hard discrepancy and underpredicted harm. Three
head seeds do not constitute independent forecast seeds; average within each of
12 localities before the fixed 3,000 paired bootstrap draws.

An independent scalar implementation checks actions, matched counts, risk bounds,
moments, paired contrasts, bootstrap intervals and gate decisions. Exact repeated
neural inference, all forest-control replay and the three prior neural prediction
hashes are required. The original task-risk prediction is the new exact unit
control; it is not mislabeled as an independent new treatment.

No threshold sweep, validation checkpoint selection or independent test use.
Results remain exposed development and cannot authorize deployment, Stage5C,
SMC, metric/seconds, true3D or foundation claims.
