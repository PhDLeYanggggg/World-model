# Pre-Readout Conditional-Prevalence Sensitivity

## Material Passport

Added after fixed training began, before any new held membership outcome
readout. This is a prospective secondary diagnostic, not part of the original
training registration and not a revision of its primary endpoint. Existing
source data have historical development exposure. No held outcome is used
to change models, labels, thresholds or stopping.

## Reason and Computation

The registered reference is an unconditional, training-only easy prevalence.
The primary readout population is conditional on positive causal forecast
disagreement. A classifier could partly gain by learning this population's
base rate. Retain the primary result, but also compute a stronger constant
using only the three fitting localities' easy prevalence within positive
disagreement, with equal-locality weighting as in the fitting diagnostics.

For a held subset with diagnostic prevalence r and training-derived constant
p, its Brier error is p*p - 2*p*r + r; log loss is
-r*log(p) - (1-r)*log(1-p), with the same numerical evaluation clipping.
The held r evaluates the fixed p; it never replaces p or enters inference.
Compare both learned arms to this constant. Average three seeds within each
locality and use the same 3,000 locality resamples and all six assignments.
Report both full and motion-only pairs. No favorable comparator selection.

This sensitivity cannot rescue a failed primary gate, certify calibration,
advance a policy, or open independent roles. It helps interpret a nominally
positive primary result. No new cost/trajectory model or metric/seconds,
human-gold, true 3D or foundation claim. Stage5C/SMC off.
