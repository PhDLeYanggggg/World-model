# Risk-Conditioned Residual Transfer

## Material Passport

Follow-up to verified ca08670b. The common-event repair completed 864 fits but
failed its scientific gate. All 50 parent artifacts and 11 source bindings
must match before this experiment. This is source-development work, not
independent calibration/confirmation. The original trajectories and all neural
heads stay frozen. No new architecture or threshold sweep.

## Hypothesis And Excluded Repeats

Does conditioning a residual on the causal scores of the model being corrected
improve transfer from inner to outer estimators? Seven context summaries alone
do not represent those score differences. Add exactly two bounded features:
predicted H / causal disagreement envelope, and predicted H_E / predicted H.
At zero denominator return zero; only floating-point boundary excursions are
clipped to [0,1]. These ratios are not calibrated probabilities.

In fitting use each bank's frozen producer scores; at inference use the frozen
original outer scores. Neither realized error nor easy membership is an input.
The common response, outer evaluation cut, fitting weights, ridge 0.1, bins,
RMS convention, rosters and seeds remain fixed. Original D/H/D_E outputs and
both trajectories remain unchanged; only H_E is shifted and clipped to [0,H].

Controls: original, previous common-event seven-feature OOF, original inner-
event OOF, prior three-locality in-sample context, new score-only, and both
matched cyclic score+context controls. Retain all variants and motion-only.
There are 144 views and 864 new candidate closed-form fits, not neural updates.

Before fitting, re-extract original-head predictions on fitting rows only.
For each of the three banks, measure location/scale differences and verify:
common-context minus in-sample-context coefficient vectors equal the fixed
ridge projection of original_fitting_score minus inner_fitting_score.
This identifies an algebraic difference, not a causal explanation. A naive
prediction-difference offset simply returns the already-failed in-sample
control. A mere matched producer swap/renamed old probe is not the experiment.

Support diagnostics retain all views: known positive-disagreement rows,
positive harm rows, score variability and feature distributions. The fraction
outside the fitting central90% is descriptive, not an OOD/safety rule. Require
finite varied score features and positive easy-harm fitting support before
the fixed fit; no data-dependent feature/threshold choice or fold deletion.

## Execution And Evaluation

Commit registration, code and protocol before support extraction. Commit the
support receipt before fitting. Freeze and commit all new predictions before
new source-held readout. Old exposed outcomes are not made independent by
freezing. Reserved selection/calibration/confirmation remain unopened.

Primary is unchanged positive-disagreement easy-harm MSE. Mechanism signal
requires six positive full-input intervals against common-event context.
Overall repair additionally requires six positive intervals against original,
old inner-event OOF, prior in-sample context, score-only and both cyclic
score+context controls. Require no negative/missing top10 or coverage intervals
in these contrasts. Secondary AUROC wins cannot replace this criterion.
This gate is not a safety/noninferiority guarantee or deployment authorization.

Average three seeds per locality, then 3000 paired resamples of four localities
per assignment, fixed seed38113. Six assignments overlap, not six independent
studies; no window-independent or multiplicity-adjusted claim. All arms remain.

Use native arm64 CPU4/interop1/workers0, per-view resume receipts and heartbeat,
10GiB reserve. CREATE check is read-only. Reuse parent checkpoints with hashes;
freshly replay support, all fits/predictions,36 readout groups and direct MSE.
Do not send raw data, caches, checkpoints or per-row predictions to GitHub.

Obs8/pred12 annotation steps, detector image pixels. No metric, seconds,
human-gold, physical-safety, true3D, foundation or submission-ready claim.
Stage5C and SMC remain off. Even a positive cost result needs trajectory utility
and independent calibration before any deployment change.
