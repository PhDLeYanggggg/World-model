# Failure Taxonomy

This is a fresh full72 development experiment, not independent confirmation.
The checkpoints and original source data are hash-verified. No winning source,
seed, margin or cutoff was selected after readout.

| Question | Evidence | Conclusion |
|---|---|---|
| Did optimization fail? |72 heads converge, max15 iterations, exact refits, losses decrease|No detected optimizer/replay failure|
| Did the change remove additive negative-to-zero harm? |9,514 additive negative raw coordinates; positive guard0, zero predicted harm/easy-harm0|The tested output removes that mechanism|
| Did known-label risk improve? |20->2 violations versus two-harm additive; original4|Yes, partial repair, not a safety certificate|
| Is cost prediction better? |Signed MSE+0.136161 vs original and+0.168307 vs additive; both CIs positive|No; training objective improvement does not transfer to this cost metric|
| Is utility better at equal counts? |+0.022976% vs original, -0.043408% vs additive|Some useful discrimination remains; not a uniform improvement|
| Does full support prove safety? |37 supported vs33 original, but11 upper violations vs7|No; gains in some groups do not cancel failures elsewhere|
| Does matching repair the risk? |93,237 selected in both; positive support27 vs33, violations13 vs7|No; count alone is not the explanation|
| Are remaining failures only unknown labels? |Nine upper failures need unknown completion; two fail on known labels|Unknown support matters more now, but is not the only failure|
| Are zero-label leaves fabricated? |174,352 repeated easy-harm leaves have zero TRAIN target and stay unchanged|No pseudolabels; no proof of zero population harm|
| Is this just an unlucky seed? |Seed17/29/43 signed MSE deltas all positive; upper violations2/5/4|No favorable-seed-only interpretation|

## Locality Breakdown

Each locality has six fixed source views, not six independent scenes. Counts
below are complete support / easy-risk upper violations / known-label violations.
Exact means and all controls are in `slices.json`.

| Locality | Original | Positive | Positive-original signed MSE | Matched utility % |
|---|---|---|---:|---:|
|007|6/0/0|6/0/0|+0.063017|+0.002198|
|008|6/0/0|5/1/0|-0.001162|+0.109837|
|020|0/0/0|3/0/0|+0.090361|0|
|048|6/0/0|4/2/0|+0.015257|+0.025254|
|067|2/3/0|3/3/0|+0.159557|+0.001638|
|074|3/0/0|3/0/0|-0.008943|+0.000609|
|082|0/0/0|1/2/0|+0.013505|0|
|110|3/0/0|3/0/0|+0.682921|+0.056272|
|112|0/3/3|1/2/2|+0.227658|0|
|119|3/0/0|3/0/0|+0.020787|+0.025638|
|124|1/1/1|3/0/0|+0.360507|0|
|126|3/0/0|2/1/0|+0.010468|+0.054267|

Zero violations with zero complete support is not success. Undefined risk and
nonpositive supported utility still fail. Utility uses full known reference
mass, not trajectory accuracy improvement. Source differences are descriptive,
not a license to deploy only the favorable localities.

## What Is Established Versus Unresolved

The earlier additive failure was not only lack of useful features. The same
features with this output/loss change greatly reduce known-label harm, yet
global cost accuracy worsens. Mean-preserving positivity is therefore not a
sufficient repair. The paired intervention changes both link and loss, so their
individual causal effects are unresolved. Projection also couples raw harm to
benefit;23,824 predicted-benefit changes were recorded.

The next diagnosis must separate total-harm, easy-harm and projected-benefit
error, selected versus unselected regions, and TRAIN harm support. A normalized
conditional objective may misweight rare/small-magnitude leaves relative to
the registered signed quadratic readout, but that remains a hypothesis until
prediction-level error decomposition. No claim that detector noise, feature
shift or reference inflation has been ruled out everywhere.

No new threshold search, future-quality exclusion, test access or deployment.
Independent roles remain closed. Stage5C/SMC off; image-local raw-stride silver
development evidence only, not physical safety, metric, seconds, true3D or foundation.
