# What Changed, What Worked, And What Remains Unproved

I tested whether learning the value of overriding an existing decision avoids
discarding useful neural predictions. The experiment is complete: 144 small
Torch cost heads, 72 ridge fits, 288 frozen policy views, three seeds and all six
source-role rotations. No new trajectory forecaster was trained.

The change substantially narrows regressions against the incumbent. The
incumbent-relative policy has all-ADE gains of -0.00893% to +1.15195%, with 33
positive and one negative source-bootstrap interval. The matched-input
floor-reference control ranges from -0.76264% to +1.68534%, with 14 positive and
five negative intervals. Both use exactly the same candidate/floor forecasts,
381 causal inputs, model capacities and training budgets.

That is useful progress, not a complete primary pass. Directly comparing the new
arms gives 19 positive and eight negative all-ADE intervals. Incumbent-relative
learning is more conservative against the original policy, but not uniformly
more accurate than relearning floor-versus-neural selection. The registered
consistent-improvement gates fail; I have not relaxed them after seeing results.

## The Directional Test Is Informative

The predeclared add-only rule preserves all original neural decisions and can
only add interventions. Its all-ADE gains are +0.00743% to +1.17500% in all 36
views, with 33 positive and zero negative intervals. The remove-only rule has
zero positive and 30 negative intervals. These rules share the same learned
scores, so this is not the result of fitting another threshold after readout.

However, add-only still has one negative hard-subset interval and a worst-locality
hard loss of 0.18781%. Its all-ADE average can conceal that failure. It remains a
promising development diagnostic, not a selected deployment upgrade or untouched
confirmation result.

## Safety And Reproduction

Both incumbent-relative and add-only have 0.0% worst positive-easy degradation
versus CV across all 36 defined views, with no observed zero-CV harm. Here 0.0%
means no positive locality-mean easy degradation; it does not mean zero individual
error or no harmful rows. The risk model remains uncalibrated: 65 of 139 supported
selected locality/views exceed 2% realized positive harm/reference cost, and 110
underpredict the ratio. The ratio reaches 23.409%; it is not net easy degradation.

All 144 checkpoint prefixes and 72 full ridge arrays replay. Every new decision
froze before readout; 144 independent coordinate arrays and 3,600 metric
reductions pass. The 379 tests in 63 scoped files pass; the full legacy suite was
not run. Training took 712 seconds and evaluation 241 seconds. Configurations,
loss traces, receipts, all negative branches and per-locality results are retained.

Deployment stays unchanged. The next minimal experiment should test incremental
additions under scene-level joint controls at matched intervention counts/risk,
and inspect the residual hard failures. This is not a reason to keep tuning
removal thresholds on the same readout or open confirmation sources prematurely.

All results are opened-development, detector-derived, image-pixel obs8/pred12 at
raw annotation stride12. The 36 views reuse twelve sources; their intervals are
conditional diagnostics, not independent confirmations. No metric/seconds,
human-gold, physical-safety, true3D, foundation or submission-ready claim.
Stage5C and SMC remain off.

See [complete results](results.md), [action accounting](changed_action_accounting.json),
[loss traces](training_losses.svg), [failure analysis](failure_analysis.md),
[reproduction](reproducibility.md) and [method positioning](method_positioning.md).
