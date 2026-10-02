# Cost-Aligned Positive Harm Learning Control

Register before real fitting. The frozen72-head diagnosis found easy-harm
overprediction tails account for90.87% of increased signed-score MSE; projection
helps and most error is unselected. This is evidence to test a loss change, not
proof it works. All12 exposed development localities and three fixed head
seeds remain; independent selection/calibration/confirmation roles stay closed.

## Fixed Form, Changed Loss

Keep the original trees, causal seven-feature quality schema, TRAIN-fitted
normalization, masks, source/whole-recording partitions, raw B/R/ER, numerical
guards, and mean-preserving positive form:
`H(q) = mu_leaf * exp(beta*q) / E_train(exp(beta*q))`.
Fit total H and easy EH only; zero TRAIN target leaves remain unchanged.
Penalty remains1. There is no cutoff/threshold search or post-hoc shrinkage.

Replace conditional relative Poisson deviance with a quadratic surrogate of
the frozen three-score training objective, plus the same beta ridge penalty.
Let sigma be the three frozen signed-score scales, and a=1/sigma_U^2,
b=1/sigma_A^2. With B/R/ER fixed at the tree leaf predictions:

`t_H = y_H + [a*(B-y_B) + .02*b*(R-y_R)]/(a+b)`

`t_EH = y_EH + .02*(ER-y_ER)`

The objective is half squared error to those two equivalent targets using
`s_H=sqrt(1.5/(a+b))`, `s_EH=sqrt(1.5)*sigma_E`, plus half ridge norm.
Its change equals the mean squared error change of the three signed scores
for a fixed leaf's B/R/ER, up to floating-point error. The equivalent targets
may be negative: they are algebraic supervised labels, never inference inputs
or negative physical harm predictions. Means for the positive link remain the
original nonnegative H/EH means, not these equivalent targets.

The cross-terms with benefit/reference labels must be retained. A scalar test
caught and repaired their omission before any real fit. This is a per-tree raw
score surrogate, not a claim of exact optimization of the projected ensemble.
Gauss-Newton with TRAIN-only Armijo search, max128 iterations and gradient
tolerance1e-7; a nonconverged fit fails explicitly. Nonconvex stationarity is
not global optimality. Record objective, gradient, mean-preservation and guards.

## Comparisons and Advance Screen

Fresh cost-head fit vs cached_verified original, additive and previous positive
Poisson heads. Recompute every old prediction/action/readout exactly; replay
fresh fits, serialization and inference. Evaluate full and same-query
count-matched intervention sets. Preserve the2% selected-positive-easy-harm
over selected-easy-reference budget and unknown-outcome completion. Undefined
and empty support do not pass. No future-quality filtering or exclusion.

Report known risk separately from unknown upper completion, all scores,
conservative utility, worst upper ratio, support and interventions. Use the
same nominal paired3000 locality bootstrap. Advance only if signed-MSE CI is
below0 and full/matched utility CI above0 versus each of the three controls,
complete support is not reduced against original, and upper violation count
and worst upper risk do not worsen. An internal pass is not deployment or
independent confirmation. Any failure stays visible.

Native arm64 local pilot, then full72 if engineering/runtime checks pass.
Checkpoint each completed head in the owned CREATE directory, read/write only
there, no remote scientific computation on login nodes. No new local numerical
cache below the fixed10GiB reserve. Resume and exact replay supported. No
change to simulation jobs or unrelated staged data. Up to12h, no speed-based
sampling downgrade. Public output cap20MiB; checkpoint cap512MiB.

Labels remain detector-silver; image-local obs8/pred12 rawstride12. No metric,
seconds, true3D, foundation, physical-safety or submission-ready claim.
Stage5C and SMC remain off.
