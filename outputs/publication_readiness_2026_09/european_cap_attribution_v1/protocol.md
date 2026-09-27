# Frozen-Cap Mechanism Attribution

## Material Passport
Fitting-development follow-up to the temporal-context screen at 9866faa4.
That study's outputs have already been examined. This analysis is registered
before new projection contrasts, not a claim of previously unseen data.
All fitted weights, normalization, nuisance producers, forecasting families,
source assignments, row support and target definitions remain fixed.
Fresh_run denotes new deterministic projection/scoring; cached_verified denotes
the parent models and forecasts. No new fitting or neural updates are planned.

## Falsifiable Question
Does the frozen predicted all-harm ceiling suppress predictive information in
the temporal readout, or does it protect against unstable residual predictions?
A large label-aware pointwise error floor alone cannot answer that question:
an expected cost is not an upper confidence bound on each realized event.

## Fixed Intervention
Recover each probe's unprojected easy-harm score from its frozen coefficients
and past-only features. Reproduce the old clipped output exactly. For each of
score-only, seven-summary, ordered-history and history-plus-neighbors, compare:

1. Nonnegative: max(score, 0), an unconstrained diagnostic upper range.
2. Frozen cap: clip(score, 0, frozen predicted all-harm), the original control.
3. Envelope: clip(score, 0, causal forecast-disagreement envelope).
4. Envelope coupled: the same easy prediction, with predicted all-harm raised
   to max(old all-harm, new easy-harm).

Signed scores are retained only for squared-error decomposition. The envelope
comes from forecast disagreement and the matched ADE triangle inequality, not
future labels. The uncoupled envelope can violate easy-harm <= all-harm. The
coupled output restores this ordering but can worsen all-harm prediction. No
arm is presumed safe or deployed. Other cost moments remain frozen; this is
not an unrestricted joint-model fit or proof of all possible moment identities.

## Exclusions and Freeze
Use the existing 432 inner fitting-locality views, six source assignments,
three cached training seeds and full/motion-only families. All nuisance models
and probe fits exclude the inner scoring locality and the outer locality.
Source localities rotate across views and are already development-exposed.
No outer conditional-head experiment, independent selection, reserved risk
calibration or confirmation outcomes are accessed. Hash-bind all 1728 recovered
score vectors before new label scoring; coefficients and inputs must match the
parent receipts. Store hashes rather than another large row cache. This permits
exact replay and resume while preserving at least 10GiB free disk.

## Registered Analyses
The primary remains expected easy-harm MSE. Retain top10 harm-mass capture,
absolute log-coverage error and all-harm MSE guards. Fourteen fixed contrasts:
envelope and coupled-envelope versus frozen cap for all four feature arms;
history-neighbor versus score, summary and history controls under both the
frozen cap and coupled envelope. No best projection, threshold or penalty is
selected using these outcomes.

Report negative-clipping and envelope-clipping error changes, cap intervention
frequency, nested-order violations, signed and projected absolute MSE, and
the exact difference (cap-y)^2-(envelope-y)^2. Split that difference into
helping/harming contributions and event/zero-event contributions for diagnosis
only. Neither these label-derived slices nor pointwise oracles enter inference.

Average three outer-context contrasts inside locality and seed, then three
seeds inside locality. Use 3000 paired resamples of four scoring localities
with seed 71429, separately for six overlapping assignments and two families.
Intervals remain exploratory and unadjusted; missing support is not dropped.

A joint conditional follow-up is supported only if full history-neighbor's
coupled envelope has >=2 positive and zero negative/missing primary intervals
versus its frozen cap, and likewise versus both matched score and summary
coupled-envelope controls, with zero negative/missing guard intervals for all
three comparisons. This is an information screen, not a deployment gate.
If it fails, do not tune another cap or these temporal coefficients. Use the
observed direction to decide whether causal observation quality, independent
event-bearing training sites or a separately justified structured conditional
model is the next bottleneck. Do not open reserved roles to resolve failure.

## Claims and Runtime
Local native arm64 Python, four compute threads, one interop thread, no loader
multiprocessing; heartbeat, exclusive lock, receipt-checked continuation.
CREATE is checked read-only and unrelated jobs are preserved. A small pilot
estimates local cost before all views run. This computation needs no new GPU
training. Configuration and aggregate evidence enter Git; source data, detailed
row outputs and caches stay private and ignored.

Detector pixels, obs8/pred12 native annotation steps, no seconds or metric
claim. No physical-safety, human-gold, true3D, foundation or submission-readiness
claim. Stage5C execution and SMC remain disabled.
