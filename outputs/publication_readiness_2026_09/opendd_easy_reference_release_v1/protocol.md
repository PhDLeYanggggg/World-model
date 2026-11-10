# Predict whether an edited recording has an easy reference

The previous OpenDD sign rule made 32 releases, of which 16 lacked
an estimable selected-easy denominator. Hold each sign nomination
fixed. Fit a separate binary head for whether either of the two
remaining edited actor-windows has positive observed CV ADE at or
below the existing donor-derived easy cut. Any valid future step
is sufficient to compute this label, as in the original evaluator.
This is a training label, not a future mask or a risk claim at inference.

For each of the existing 63 location-pair and seed cells, train on the
same five donor locations. Candidate generation also excludes the
donor itself. The donor anchor is the previous deterministic causal
anchor; calibration and held use the frozen additive anchor. Train
200 updates, batch size 160, with the original balanced-by-donor
sampler, four native CPU Torch threads and zero workers. The features
are symmetric in the retained and new recipients: motion magnitudes,
CV/OLS4 differences, CV relative to the observed query footprint and
roster size. No labels enter scoring.

Compare the original sign rank, sign multiplied by predicted reference
probability (the primary variant), and the reference probability alone
(ablation). The product is a ranking score, not a calibrated joint
probability. The three arms use the same two edits when they release,
but may release different recordings. The existing tied-prefix search
requires the original 2% checks, beating both motion baselines and
improving on the additive incumbent in calibration. Release only at the
frozen threshold; otherwise use CV. An empty release is not a risk pass.

Freeze scores before calibration and thresholds before held readout.
Require exact reproduction of the old sign policy. Report support
Brier and AUC, undefined-risk counts, strict passes, and the paired
conservative effect on CV improvement with 3,000 location resamples
after averaging seeds and calibration roles. AUC alone is not success.
To justify advancing, the primary variant should find more strict
releases, have no measured or undefined risk violations, and improve
over sign with a positive paired lower interval. This is development
evidence on previously analyzed OpenDD locations, not an independent
test. No external confirmation outcomes or thresholds change.
