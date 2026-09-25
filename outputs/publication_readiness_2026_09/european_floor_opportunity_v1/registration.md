# Neural Opportunity Above a Protected Damping Floor

Registered before this diagnostic readout, 2026-09-25. This is an opened-source
development analysis, not a new independent test, training run or deployment.
Both completed cross-moment modes (batch, fitting) remain in the report. No
winner, threshold, forecast, risk budget, fitting split or calibration is selected.

## Question

The earlier opportunity study used CV as reference and different producer
support. Here the reference is equally protected damping from the current
four-fit/eight-complete-chain-excluded protocol. Is neural failure due to little
incremental candidate value, poor intervention choices, or reverting to CV when
damping would have helped? Preserve all three seeds, three folds and both event
heads. There are 18 paired groups per mode, not 36 independent experiments.

## Frozen Accounting

Let c be CV ADE, d the protected-damping policy ADE, n the neural candidate ADE,
and s the saved causal neural switch mask. Original policy is s*n+(1-s)*c.
The offline rebased policy is s*n+(1-s)*d. No s is changed. This rebase was not
calibrated against d, so favorable results do not permit deployment.

Reachable neural benefit is max(d-n,0). Report captured benefit, missed benefit,
selected harm, CV fallback regression and CV fallback relief. Original gain is
captured-harm-regression+relief; rebased gain is captured-harm. Both are divided
by locality sum(d), then averaged equally across the fixed eight localities.
Retain missing and zero-denominator localities as undefined, without epsilon.

Two unavailable future-label diagnostics are kept separate: min(d,n), and
min(c,d,n). The latter includes the value of undoing harmful damping, not only
neural value. Original policy can outperform min(d,n), so its deficit is signed.
Oracle decisions use ADE; FDE evaluates that same chosen forecast, not another
FDE oracle. Neither oracle is an inference input or deployable policy.

Reasons are recomputed from saved utility, event mass, risk, observed motion
and fitting support before outcome access and matched to frozen decision hashes.
Future labels only quantify missed benefit afterward. Ordered gate attribution
is descriptive accounting, not unique causal mediation.

## Support and Uncertainty

All, easy, hard, complete, partial, endpoint, easy-complete and hard-complete
subsets are retained. Original all-row population includes unknown labels; these
are not zero-cost successes. Completeness is evaluation-only, never a filter for
inference or replacement of the primary population. Easy/hard cutoffs remain
fitting-only. Zero-CV harm is reported separately from positive-easy degradation.

Three seeds and 3,000 paired locality-bootstrap draws (seed39271) per view.
Intervals are conditional, dependent, unadjusted development diagnostics; no
overlapping-window independence, population guarantee or test-set promotion.
Check original parent metrics, coordinate errors, exact conservation and bootstrap
reductions separately. Store resumable per-group results and heartbeat locally.

## Execution and Boundaries

Local native arm64 CPU4/inter-op1/workers0. No new training or remote job. Keep
10GiB free; stop only for hard failure, not slowness. Hash both predecessor
analyses, code, configuration, role lineage, heads and decisions. Push this
registration before the new readout. Full predictions and group archives stay
local; only aggregate summaries and source code enter Git.

EuropeanSquares released detector tracks: image pixels, obs8/pred12, rawstride12.
No metric, seconds, human gold, physical safety, true3D or foundation claim.
Historical Stage37 is not recertified. Reserved roles stay closed. Stage5C and
SMC remain disabled. A follow-up model may be designed from this development
diagnosis but must not be relabeled independent confirmation.
