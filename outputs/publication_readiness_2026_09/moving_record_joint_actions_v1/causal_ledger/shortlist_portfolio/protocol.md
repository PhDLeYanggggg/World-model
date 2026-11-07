# A Fixed-Budget Portfolio For Opening Action Retrieval

Test whether different causal cost rankings preserve more useful
actions than family diversity alone. The forecasts, native 8/12
annotation-step task, direct interventions, exposed Moving A cohort
and eight-slot budget are unchanged. The first slot is the old joint
support winner. Cycle through joint support, predicted lower gain,
predicted lower gain minus positive all/easy excess, and negative
positive excess, breaking ties by joint score and stable row order.
Prefer an unused agent whenever available; never repeat an action.
Fill to the same count as the family-diverse control.

Use unchanged action features and the same five-target ridge
objective, source/record weights, and prefix weighting on the new
shortlist. Fit on seven sources using donor-excluded fitting
shortlists, calibrate on six sources, hold one source out, and repeat
for each of the three original forecaster seeds. Compare the
risk-penalized winner and the useful-event winner of this one refit.
Calibrate each on the original donor grid, without adjusting the risk
or support criteria. Allow at most one additional opening before
direct's first switch. No new forecaster is trained.

Report positive supported first-two-query recall on the 18 original
zero-direct views, including overlap/gains/losses vs the exact old
shortlist. Retrieval succeeds descriptively only if it retains a
useful action in more views; this is not a model utility or safety
test. The policy screen separately requires more complete passes, no
additional signed easy violations, and a strictly positive paired
95% interval for calibrated completion-lower utility versus the
unaugmented family-diverse selector. Also report diagnostic results
at identical queries and switch counts. Bootstrap 3,000 source
resamples with three seeds averaged within each source.

Inference uses no target arrays. Freeze decisions before evaluation.
No label-aware choice, identity-fallback certificate, independent
outcomes, deployment change, Stage5C, or SMC. This is a development
ablation, not independent confirmation.
