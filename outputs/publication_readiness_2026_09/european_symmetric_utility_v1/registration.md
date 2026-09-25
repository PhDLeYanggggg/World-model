# Symmetric Utility, Fixed Risk

Registered before new fitting or readout. Source-development experiment only.
The opportunity diagnosis suggests that asymmetric utility regression can
overestimate harm when the prediction is later treated as an expected cost.
The analytic example is not evidence that a real-data repair will work.

## Single Factor

Change the utility head objective from underharm4 to symmetric MSE for both
the frozen neural trajectory candidate and fixed causal damping 0.97.
Keep all 72 event-risk heads frozen, including their asymmetric loss; their
outputs, 2% predicted-risk budget and both source-support controls are unchanged.
No threshold, forecast, schema, teacher, baseline, feature, label, split,
normalization, sampling sequence, or optimization-budget change is permitted.

Three seeds (17/29/43), three outer source folds, two candidates: 18 new Torch
utility heads, each 2,000 updates; total 36,000 updates. Width64, batch256,
learning rate0.0003, gradient clip5; checkpoint and heartbeat every200 updates.
A 100-update real pilot is resumed inside the first fixed budget, not additional
training. Every fit completes before any outer-readout policy comparison.

## Fixed Readout

Retain all 48 candidate/event/risk-head/guard views and five joint controls
per view. Compare paired new-vs-old utility policies and neural-vs-protected
damping. Report full and joint populations separately; actual matched counts
apply within candidates, not across candidates. Retain undefined contrasts.
Report all/easy/hard ADE, supported FDE, worst locality, tail errors, zero-CV
harm, switch rates and 3,000 paired source-locality bootstrap resamples.
No winning seed, threshold or view will be selected from this readout.

Verify all 18 checkpoints by exact replay of 4,096 held rows per head; compare
preprocessing, initialization constants, fitting draws and sampler RNG to the
corresponding frozen underharm4 head. Verify all current and old artifact hashes.
Observed safety requires worst-locality positive-easy degradation <=2% and
zero added error on zero-CV rows. Four such rows in one locality are insufficient
for a general safety claim; the joint pilot contains none.

## Evidence Boundary

Only the 12 opened EuropeanSquares source-development localities are used.
318,969 targets; unknown future labels remain in inference and are excluded
from supervised losses. Reserved model-selection, risk-calibration and final
confirmation groups remain closed. Three seeds and conditional locality CIs
do not turn reused development data into independent confirmation.
Released detector tracks, image pixels, obs8/pred12 at raw stride12; not t+50,
seconds, metric, human gold, physical-safety certification, true3D or foundation.
No new trajectory/JEPA fitting, deployment promotion, Stage5C or SMC execution.
