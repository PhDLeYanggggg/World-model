# Why Risk Conditioning Was Insufficient

## Verified Findings

The omission of model risk scores was a testable input gap, not a proven root
cause. Fitting-only inference established nonconstant score features, matched
row alignment and measurable differences between the inner and outer models.
All 432 coefficient differences equal their fixed ridge projections to
numerical precision. However, adding those scores did not produce a stable
full-input MSE gain against either the original or common-event control.

This is not an import failure, interrupted fit, missing-feature placeholder or
changed test target. All 144 views and 864 fits completed; held target and row
hashes match the prior experiment. Unknown labels remain excluded. All original
D/H/D_E predictions are unchanged; H_E stays within [0, original H].

The full-input comparison with common-event context has four positive and two
negative point estimates, but six overlapping MSE intervals. Its tail harm
capture has two negative intervals and its coverage-error reduction has one.
Thus a modest average-point recovery cannot be substituted for the registered
primary and guard criteria. The original-model comparison has six negative
point estimates. A wider interval crossing zero is not evidence of safety.

The motion-only arm improves against a weaker previous correction on three
intervals, but only one against original. It remains worse than score-only at
every point estimate. Neither switching to this arm nor to a favorable cyclic
control after readout is an authorized selection rule.

## Limits Of The Diagnosis

The fixed additive bins cannot establish which representation a richer model
needs. Score distributions differ, but the descriptive fraction outside the
inner central 90% is not an OOD test or a safety guarantee. Positive fitting
row counts do not measure independent scene support or statistical power.

The original all-harm output H remains an upper bound on the corrected H_E.
Whether this constraint or weak transferable conditional signal limits recovery
has not yet been quantified in this study. It would be incorrect to blame
that bound, sparse scenes, insufficient model size or training duration as
the established explanation without an additional controlled diagnostic.

Post-freeze [error accounting](error_accounting.md) separates the cross term
and correction energy exactly. This is an algebraic description of observed
error, not causal attribution. It does not supply a future-safe or deployable
shrinkage factor, and no such factor is fitted from held labels.

For the 72 dependent full-input OOF score-plus-context views versus original,
39 improve, 14 worsen with a non-helpful aggregate direction, and 19 worsen
despite a helpful direction because correction energy dominates. Against
common-event context the counts are 38 / 21 / 13. Motion-only versus original
is 41 / 16 / 15. These are dependent seed/locality views, not independent
success probabilities. Both direction and size errors remain, so a blanket
shrinkage adjustment is not an established solution.

## Next Test

Do not repeat label-cut, score-bin or held-optimal threshold sweeps. On fitting
localities only, quantify the best feasible H_E error under frozen H and causal
envelope bounds, explicitly labeling any realized-label projection as an
offline diagnostic, never inference. Compare that constraint floor with
unexplained conditional error and existing producer-support evidence. Check
the prior attribution studies to avoid rebranding their substitutions as a
new experiment. Only a differentiated finding should trigger new joint-cost
training or a representation change.

The current deployable behavior stays unchanged. Independent selection,
reserved calibration and confirmation remain unopened. These are source-
development results, not cross-dataset success or world-model dynamics lift.
Obs8/pred12 annotation steps, detector pixels only. Stage5C and SMC remain off.
