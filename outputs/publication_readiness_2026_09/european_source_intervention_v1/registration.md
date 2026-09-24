# Nested Source-Only Gain, Harm and Joint Intervention

2026-09-24. Fixed while the eighteen source forecasting fits are running, before
their predictive readout. This is development on the twelve admitted training
localities only. The six selection, twelve calibration and six confirmation
groups remain closed. It does not revise the completed predictor experiment.

## Nested Supervision

For excluded group A, use the B-only predictor for costs on C and the C-only
predictor for costs on B. Neither predictor nor its learned preprocessing may
have fitted A or the row it predicts. Fit the cost head on these B/C costs; its
candidate on A is the B+C predictor. The four-versus-eight-site producer shift
is a known limitation. Never replace source-excluded costs with in-sample ones.
Use each outer complement's training-selected strongest causal baseline as the
reference, including on its nested fitting rows. Its selection uses only B/C.

Targets are positive improvement and positive added masked ADE, in native image
pixels, with unknown labels masked rather than filled with zero. Features are
observed ego/neighbor histories, masks and forecast disagreement. Scale position
features by the maximum observed ego/valid-neighbor norm, lower bounded by one;
retain its logarithm. Standardization and cost scaling use supported fitting
sites only, with equal-site weights. No site ID, future-valid mask, endpoint or
future-derived class enters inference. Baseline/candidate rollouts are past-only.

Compare fixed ridge penalty 0.1 against a 64-wide gain/harm network trained for
2,000 updates per excluded fold and seed 17/29/43. The neural objective weights
underestimated positive harm fourfold. Reuse the tested atomic checkpoint,
sampler, optimizer and RNG resume path. No early stopping, hyperparameter search
or reserved-data calibration. Finish all eighteen cost heads before readout.

## Controls and Query Budget

On the entire source-excluded target cohort report the fixed baseline, neural
candidate and positive-predicted-net-gain pointwise rule. For scene controls,
select 96 query frames per locality by a fixed hash of recording index/frame
before reading candidate or target errors. Keep all indexed targets at each
selected query, including unknown supervision. This is a smaller *registered
joint-control population*, not the full source benchmark; all its arms use the
same rows. It is not an outcome-filtered easy/hard cohort.

Compare baseline, uncontrolled neural, pointwise, budget-constrained independent,
whole-query uniform, joint optimization, and unary-geometry/joint controls at
the independent policy's exact intervention count. Eligibility uses moving past
history and strictly positive predicted net gain. The conservative arms share
the same predicted positive-harm budget: 0.02 times the fitting-only equal-site
mean baseline ADE. This is a predicted-risk constraint, NOT a guarantee of 2%
realized easy degradation or zero-reference preservation. The original actual
easy/zero gates remain unchanged and are evaluated, not assumed.

Build edges within three median current box widths; future closeness proxy uses
half a median box width, weight 0.1. These are image-overlap proxies on forecasts,
not collision labels or physical distances. They are fixed before readout.
Use the existing scaled-risk solver with original-unit and primal/dual checks;
nonoptimal/invalid cases fall back and remain counted. Matched comparisons are
valid only for queries with certified equal counts, separately reporting zero
counts, failures and full-population fallback behavior. No label-based matching.
Only complete-history targets are decision nodes; short-history visible agents
remain context in source storage but are not controlled by this experiment.

## Readout and Limitations

Report all seeds and equal-locality paired ADE/FDE, 3,000 locality bootstrap
resamples, positive-CV-q25 easy and CV-q75 hard using fitting-only cutoffs, exact
zero-CV absolute harms, tails/worst locality, switching, predicted feasibility,
matched coverage and predicted overlap proxy. Preserve unknown outcome counts.
Three seeds are separate fits; error averaging is not an ensemble. Confidence
intervals are conditional exploratory source evidence, not final independent
test confidence. Report learning error and signed-gain correlation as diagnostics.

No risk threshold selection on these outputs, no promotion, no physical or
seconds claims, Stage5C or SMC. Negative results motivate a new registered
source-only experiment; they do not authorize opening confirmation data.
