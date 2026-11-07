# Direct Neural Increment Over The Selected Motion Forecast

On the fixed exposed Moving A roster, keep the CV fallback, OLS4
comparison, the pure-motion joint recipients and every query's
intervention count. Compare retaining each selected motion path,
replacing it when the original CV-relative critic predicts a positive
increment, and replacing it when a new direct critic predicts a
positive increment. Only the three original neural paths are offered
as replacements. A path identical to CV is not eligible to replace
an existing switch. Recipients never change.

The new critic uses the same causal descriptor construction, with
the selected motion path as the reference, and the same source-balanced
ridge penalty of 0.002 (no penalty on the intercept). Fit one head per
neural family. For a known future, the target is the available-label
ADE of the selected motion path minus that of the neural path. For
a wholly unknown future, use the negative maximum pointwise
disagreement of the two paths. Divide both by the original past
motion scale. Keep partial rows as observed rather than imputing
their missing steps. Do not change the official scoring estimand.

For each outer held source, choose the motion anchors on each of
the other 13 sources with a critic fit on the remaining 12 sources.
Fit the incremental critic on these donor-source actions and targets.
The outer source's outcomes are not used for its anchors, features,
heads, or selection. Reuse the same frozen three forecaster seeds.
The fixed zero threshold is not selected on outcomes. Score only
after every output has frozen. All results are exposed development.

The primary comparison is the direct increment policy versus the
original CV-difference policy at exactly the same recipient and
switch counts. Also compare both to retaining the motion anchors.
For each, separately apply the existing predicted CV screen to
its chosen paths, using the original CV costs and 2% rule. Such
screened arms may have different counts and are not a matched
utility control. Report paired 3,000-draw source intervals after
seed averaging, available ADE/FDE support, original recording
risk passes and relative-path bounds. No threshold tuning,
independent labels, change to the running calibration candidate,
deployment, Stage5C or SMC. Both incremental scores are clipped
to the same signed maximum path-disagreement bounds before ranking.
