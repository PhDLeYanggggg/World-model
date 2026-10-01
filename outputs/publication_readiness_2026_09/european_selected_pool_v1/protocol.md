# Frozen Selected-Pool Accounting

## Material Passport

Experiment mode: run/validate. Registered follow-up to component calibration.
All outcomes here are previously exposed development evidence. No new parameter
fit, policy search, independent confirmation, or deployment change is allowed.

## Question and Existing Assets

Does component calibration worsen error estimation on the set it retains within
source recordings, or does miscalibration appear mainly after locality transfer?
Do not assume either mechanism. The previous selected-risk diagnostic did not
support greater average optimism for selected versus unselected rows under an
older learned policy. The subset-excess training also failed risk gates. Neither
is this fixed-calibrator, source-OOF versus transfer accounting control.

Reuse all 72 parent heads and calibrators, all 216 transfer views, all three
head seeds, the original source optimization/validation partitions and frozen
causal actions. Source OOF excludes each entire validation recording from its
own margin fit. Add source full-fit resubstitution only as a labeled diagnostic
to expose OOF-to-final-coefficient differences, not as independent validation.

## Fixed Calculations

For harm-only, reference-only and joint arms, partition raw eligibility into
retained and removed actions. Validate exact action/prediction hashes against
the parent. Compute benefit B, positive harm H, reference R, easy-reference ER
and easy harm EH, plus known/unknown counts and causal disagreement envelopes.
No future label creates action eligibility. Unknown outcomes are not zeros.

Primary diagnostic is the within-view change in envelope-normalized signed
easy-risk prediction bias, retained minus raw, using the same adjusted
predictions on both pools:
`bias(A) = sum_A[(EH - predicted_EH) + .02*(predicted_ER - ER)] / sum_A envelope`.
All sums in this bias use native known-label rows, not complete-case-only rows.
Unknown selected occurrences and their envelopes are separately reported.
Also report all-risk bias, actual selected ratios and retained mass fractions.
Zero denominators are undefined; report coverage, never silently impute a pass.
Record calibration support per recording. As a labeled secondary diagnostic,
repeat the bias contrast restricting the raw pool to recordings with supported
coefficients. This separates unsupported-calibrator rejection from selection
within a supported pool; it does not change any action or future-label mask.

Verify the exact identity for each defined all/easy ratio:
`risk_kept - risk_raw = (H_kept*R_removed - H_removed*R_kept)/(R_kept*R_raw)`.
It holds without assuming removed-reference positivity. This is arithmetic,
not an uncertainty bound or causal explanation of domain shift.

Report source OOF, source full-fit resubstitution and transfer separately.
Pair each target view to its source/head calibration. Compare raw-pool and
retained-pool bias, and the OOF-to-full-fit change. Domain, data mix and sample
size differ; these contrasts do not identify a causal domain-shift effect.
Nominal 3,000-draw intervals average dependent views within locality first.
Report recording counts and per-head results; no best-seed selection.

## Evaluation and Decision Boundary

Describe all arms, not only the six known transfer failures. Explain whether
within-source selection bias increases, whether source evidence is too sparse,
and which failure patterns also occur after transfer. Positive or negative
effects both remain results. Do not retrofit thresholds, margins, labels or
the unchanged 2% selected-reference risk contract. Any next training design
requires a separate frozen registration, not a claim of success from this audit.

Obs8/pred12, stride12 raw frames; image-local detector-silver. Independent
selection/calibration/confirmation stay closed. No metric, seconds, human-gold,
true3D, foundation or physical-safety claim. No Stage5C execution or SMC.

## Resource and Reproduction

Local native arm64, CPU4/interOp1/workers0. Pilot one real source/head group;
retain 10GiB disk reserve and stream compact scalar aggregates. Save immutable
per-group private diagnostics for resume, not new trajectory caches. Recompute
all groups and transfer views for exact replay. No CREATE job is needed if the
pilot fits locally. An existing completed remote diagnostic is not a live job.
Run scoped arithmetic/leakage regression tests. Full legacy suite is not implied.
