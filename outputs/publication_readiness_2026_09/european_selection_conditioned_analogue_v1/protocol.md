# Fixed-Policy Reference Conditioning

## Material Passport

Mode: authorized experiment execution and validation. Previous turn: progress,
all72 pooled-reference views/replay completed; coverage repaired, fitting
information screen passed, risk magnitude failed. Current hypothesis: using
all eligible reference labels for queries selected by a low-risk policy
contributes to event-frequency and harm-size mismatch. This is a hypothesis,
not an established explanation or a new neural architecture claim.

## Single Changed Factor

Restrict supervised reference donors to rows selected by the SAME frozen
original head/policy in the same context and head seed. Query selection is
unchanged. Exclude the query's complete source-qualified recording as before.
Donor feature normalization remains exactly the full eligible bank, including
unknown feature rows. Do not restandardize on selected donors. Labels remain
H/envelope and easy-H/envelope; raw coordinate error is never pooled.

Retain k32, Euclidean distance, 380 causal features, std floor1e-6, clip8,
chunks256, uniform averaging, paired within-recording shuffle seed20261029.
Selection changes which donor labels are shuffled, so permutations differ
between banks; this is reported, not an identical-permutation claim.
Retain all6contexts x3headseeds x4sources =72views and the fixed62,902 query
occurrences, including unknown outcomes. Insufficient selected donors remain
unsupported with undefined primary MSE, never fall back to unselected donors
or silently delete views. The real pilot tests engineering, not scientific lift.

Eight controls: original, estimation continuation, full-bank mean/nearest/shuffle,
and selected-bank mean/nearest/shuffle. Full-bank predictions and scores must
exactly reproduce pooled v2. Copy B/R/easy-R; replace H/easy-H only. Count joint
envelope violations, without posthoc projection or new policy execution.

## Registered Readout

Primary descriptive magnitude screen requires all72 views supported, selected
nearest easy signed-risk MSE below BOTH original and full-bank nearest, all-risk
MSE not worse than full-bank nearest, and at least12/18 four-source context/head
seed pairs improving easy MSE over BOTH controls. Slack1e-12, equal-view average,
same source-normalized signed-risk definition. No support-based roster change.
Passing would motivate a separate fitted-model experiment, not authorize a new
held readout or deployment. Report every failure and the complete eight-arm table.

For both full and selected nearest neighbors, record event probability as the
neighbor event fraction and conditional severity as mean positive harm fraction.
Their product must equal the nearest mean fraction. Score Brier, event/non-event
harm SSE, predicted harm mass on events/non-events, and conditional severity SSE
only where event-positive queries have positive donor support. Zero-event donor
sets are explicitly counted. Query outcomes enter these diagnostics only, not
reference normalization, selection, distance or prediction. No oracle event or
severity is deployed. Full/selected donor event counts and source composition
are recorded. These diagnostics separate arithmetic effects, not causal proof.

No fitting CI/p-value or independent significance claim. Head seeds and repeated
occurrences are not independent forecaster replicates. Retain source/context/seed
slices descriptively, without selecting the best source, arm or threshold.

## Reproducibility and Boundary

Bind protocol, source, tests, prior-control reconciliation and parent registration/
complete/summary before real execution. Native arm64 CPU4/inter-op1/workers0;
real four-source pilot, full72 views, exact replay and separate receipt arithmetic.
Atomic per-view receipts, context/seed resume, PID/heartbeat,10GiB reserve,12h
boundary. Use authorized read-only CREATE TRAIN packets locally; no new job.
Only owned code/reports/light metrics enter Git, not data/cache/weights/images.

Upstream preprocessing, models and original selection have TRAIN exposure;
reference exclusion is not whole-pipeline OOF. No inner-held forward or donors,
outer development or independent selection/calibration/confirmation role.
Obs8/pred12 rawstride12, detector-silver pixel/local,2% positive selected
harm/reference unchanged. No metric/seconds/gold/physical-safety/true3D/foundation
claim. Stage5C/SMC disabled. Research and submission goals remain incomplete.
