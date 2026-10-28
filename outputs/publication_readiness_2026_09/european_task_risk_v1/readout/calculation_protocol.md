# Seven-Arm Frozen Readout

This implements the already frozen training protocol, before any new development
predictions. Freeze and verify all216 local final checkpoints first. Compare
original, additive, Poisson and cost forest controls, contemporary quadratic,
task-utility, and task-risk. Replay strong-control predictions/actions exactly.
Verify every input hash, source identity, TRAIN preprocessing, known-query draw
hash, causal eligibility hash, objective, step count and final checkpoint bytes.

The candidate is task-risk. Its six primary contrasts are against quadratic,
task-utility and original, each full and per-query minimum-count matched policy.
Rank matched interventions by predicted causal benefit minus harm and stable row
ID, never future error. The denominator is the same full known reference-error
mass. Common unknown interventions cancel in paired bounds; exchanged unknown
interventions retain their deterministic envelopes. Do not remove unsupported
rows or localities, or credit empty selections as safe.

Use the unchanged hard gate: causal motion/support and predicted benefit greater
than harm, predicted harm at most2% of reference, predicted easy harm at most2%
of easy reference. Training's smooth gate is diagnostic, not a calibrated
probability. Report its mean and0.5-threshold disagreement with the actual hard
gate, final duals/caps, training monitor, selected harm underprediction, tails,
positive easy risk and completion bounds. The diagnostic0.5 does not change any
deployment decision.

Average repeated views within each of12 localities, then bootstrap paired
localities3,000times with seed20261028. Preserve nominal95%intervals, rather than
treating rows/head seeds as independent. Every primary lower bound must be
positive, and all candidate full/three primary matched policies must have finite
complete support with no known/completion-upper2%easy-risk violation. Secondary
MSE improvement cannot override these gates. An independent scalar calculation
checks actions, matching, scores, harm accounting, bounds and summary gates.

These remain exposed source-development results. No independent selection,
calibration or confirmation labels are opened. No checkpoint or threshold is
chosen from the readout. Training completion and passing implementation tests
are not scientific success or permission to deploy. Stage5C and SMC remain off.
