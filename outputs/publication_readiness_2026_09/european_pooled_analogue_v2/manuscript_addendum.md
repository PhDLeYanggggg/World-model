# Exploratory Manuscript Note: Coverage and Magnitude Are Separate

We tested whether reference support explains the failure of a recording-excluded
causal analogue risk estimator. The reference bank was expanded from one source
to the four approved fitting sources sharing a predictor/controller context.
Query selection, the 380-feature representation, 32-neighbor rule and original
neural controls were held fixed. Query recordings were excluded from reference
normalization and supervised labels. Targets were harm fractions of a causal
prediction-disagreement envelope, not pooled raw-coordinate errors.

The expanded bank supported all 62,902 repeated query occurrences across 72
fitting views, compared with 52,654 occurrences and 24 complete views previously.
Nearest-neighbor easy signed-risk MSE was 0.004741, compared with 0.005577 for a
pooled mean and 0.011334 for matched shuffled labels. Twelve of eighteen
context/head-seed pairs favored nearest neighbors over both controls, satisfying
the predeclared descriptive information criterion.

Nevertheless, easy-risk MSE remained 15.32 times that of the original risk head,
and predicted easy-harm mass was 7.30 times observed. Overall risk MSE also
worsened relative to the pooled mean. On the same 24 previously supported views,
pooling reduced nearest-neighbor easy MSE by 27.99%, but did not close the gap to
the original model. These results distinguish reference availability and feature
information from accurate conditional magnitude estimation. No new policy was
evaluated or promoted.

These are exposed-TRAIN fitting diagnostics, not independent transfer results.
Upstream model and selection exposure prevents a whole-pipeline out-of-fold
interpretation; repeated occurrences and one permutation seed do not provide
confirmatory uncertainty. Undefined rank metrics and adverse sources remain
in the record. The study uses obs8/pred12 with raw stride12 and detector-silver
pixel/local annotations. It supports no metric, seconds, physical-safety,
human-gold, true3D or foundation claim. Stage5C and SMC are disabled. This is an
exploratory note, not a completed manuscript or submission-readiness assertion.
