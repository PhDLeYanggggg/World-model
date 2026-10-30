# Position-shift error geometry (post-hoc)

This arithmetic was defined after the parent negative result and registered before its own values.
No models,policies,thresholds or scientific primary change. It is not ADE and does not train an estimator.
Fresh24-group accounting:23.248551s;separate full replay:22.637795s.
All24 original parent results/history controls and all24 diagnostic groups agree exactly.

For raw error r and fitted-minus-raw position shift d,
`||r+d||^2 - ||r||^2 = 2*r.dot(d) + ||d||^2`.
Each term uses frozen causal row scale squared,available-step means within rows,known-row means within views,
then context averaging within source and macro averaging over12 sources. Unknown rows are not zero.

|Window|Squared-error change|Cross term|Squared shift|Known row views|Unknown row views|Increased-error known row views|
|---|---:|---:|---:|---:|---:|---:|
|4|0.140734633|0.132167880|0.008566753|292694|6326|172796|
|8|1.423356139|1.204155622|0.219200516|292694|6326|197075|

All12 source means of both cross term and direct error change are positive for both windows.
The shift tends to reinforce the existing error on these observed labels,not merely add its nonnegative squared length.
This is error accounting,not a causal decomposition of measurement noise or motion. The noise/process split is not identified.

## Past-Only Geometry

For an exact quadratic history with second difference a,the fitted-current minus last-position shift is
`-a*(k-1)*(k-2)/12`: -0.5*a forOLS4 and -3.5*a forOLS8.
This establishes a possible lag mechanism under that synthetic model,not evidence that the real data have constant acceleration.
On real gated predictions the shift opposes the last past second difference in:

- OLS4:256750/298325 active reused row views;
- OLS8:193275/298325 active reused row views;

Past second differences in detector tracks are not verified physical acceleration. Neither their direction nor future label errors identify true motion versus observation noise.
Per-step support and arithmetic terms remain in all24 group receipts. Different masks prevent treating per-step differences as matched horizon effects.
Floating identity residual checks scale with floating-point arithmetic only;they do not relax a risk inequality.
Wholly unknown labels remain absent. This diagnostic uses future labels only in error accounting,never as model inputs.
No optimizer updates,no independent evidence,no deployment,no Stage5C,no SMC. Parent conclusion remains negative.
