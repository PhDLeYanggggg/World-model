# Fixed Risk-Conditioned Cost Probe

Let H be predicted positive excess error over the frozen reference, H_E the
predicted easy-event harm, and B the causal disagreement envelope. Both
trajectory candidates use only observed history; B does not use a future target.
Two new inputs are H/B and H_E/H, with zero at a zero denominator. They are
bounded scores, not calibrated failure/easy probabilities. D predicts reference
error and D_E its easy-event-weighted moment; neither is modified here.

For each registered producer/controller assignment, seed and excluded outer
locality, fit on the remaining three controller localities. In the OOF bank,
each row is scored by an inner head trained on the other two fitting
localities. Two cyclic banks supply matched in-sample controls. All inner
heads were trained previously and are frozen. The common target uses the
three-locality meta-fitting easy cut; the inner heads keep their original
two-locality event definitions.

The response is (realized H_E - predicted H_E) divided by a fitting-only RMS.
Each feature uses weighted fitting terciles plus a missing-value bin. Ridge
strength is fixed at 0.1 with an unpenalized intercept. The two arms are:

1. risk_only: the two causal model scores;
2. risk_context: those scores plus seven fixed motion/neighbor summaries.

At inference, the probe receives causal features and the frozen original
outer head's own scores. Apply its predicted shift only to H_E and clip to
[0, original predicted H]. No realized error, easy label, target endpoint,
outer-locality normalization or threshold selection is used for inference.
Every prediction is frozen before new source-held readout.

This is a model-aware additive correction, not a larger trajectory model or
the full multiaccuracy/multicalibration algorithms. Adding two score features
also changes the fixed design dimension; the experiment tests that entire
registered addition, not a universal causal effect of model confidence.

Before fit, subtracting the prior in-sample and common-event context coefficient
vectors recovers the ridge projection of outer-minus-inner fitting scores.
This exact identity establishes an implementation distinction. Simply replacing
inner scores with outer in-sample scores, or adding their prediction difference
to the residual, collapses to an already-tested old control and is not a repair.

The primary target is H_E MSE on positive-disagreement known-label rows.
AUROC, top-tail capture and coverage-ratio error are retained; no ranking gain
can replace the predeclared MSE comparison and guards. The registered gates
fail. These are dependent source-development results, not independent scene
calibration, policy utility, trajectory improvement or world-model success.

All data remain detector pixels and native annotation steps (obs8/pred12).
Stage5C/SMC, physical-safety, metric/seconds and foundation claims are excluded.
