# Local Cost Estimator: Diagnostic Model and Data Card

Status: complete fitting-only experiment; primary magnitude screen failed.
This is closed-form local risk regression, not newly trained neural trajectory
dynamics. Existing neural weights and original causal policy are frozen.

Inputs: 380 past-only history/neighbor/baseline descriptors; causal disagreement
envelope; fixed source training scale; frozen original benefit/reference outputs.
Donor-only known outcomes supervise fitting. Query labels score results only.
Full source-qualified query recording is excluded from both normalization and
supervised donors. Selected donors come from the original policy, not an oracle
harm/easy label. Unknown queries remain; unsupported queries would be undefined.

Four same-context original-TRAIN inner-fit sources are pooled. Inner-held is
discarded before forward; outer development and independent selection,
calibration and confirmation stay unopened. Original upstream training and
preprocessing exposure remain: recording exclusion is not whole-pipeline OOF.
The 62,902 occurrences across 72 views are dependent; three head seeds are not
three new end-to-end forecast trainings. No independent CI is reported.

The objective fits source-balanced squared signed-risk error using 32 frozen
neighbors. Its reference offset uses only donor outcomes and frozen predictions.
Fractions are bounded in [0,1], but independent all/easy fits do not guarantee
joint moment consistency: EH>H occurs 9,019 times. The estimator is not suitable
for deployment. The no-offset weighted-harm control has no such violation but
also does not beat the original head's easy MSE.

Evaluation remains equal-view normalized signed-risk MSE, with separate
normalized harm mass, event/non-event SSE, false-safe counts and support
diagnostics. None is a trajectory ADE/FDE improvement or a physical-safety
certificate. The 2% budget means positive selected harm/reference, not net
whole-easy degradation. No policy execution or new risk calibration occurred.

Native arm64 CPU4/inter-op1/workers0; atomic receipts, heartbeat, process lock,
12h boundary, 10GiB reserve. No new CREATE scheduler submission; existing
authorized TRAIN material is read-only. Public outputs contain aggregate
statistics, hashes and code only; raw features, caches and figures remain private.

Obs8/pred12 rawstride12, detector-silver pixel/local. Not metric, seconds-level,
human gold, true3D, foundation or submission-ready. Stage5C/SMC disabled.
