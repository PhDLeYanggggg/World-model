# Input Support Repairs Reduce Some Harm, Not the Forecasting Gap

## Material Passport

`fresh_run`: 72 frozen-model/treatment evaluations (18 existing predictors,
four treatments) and a descriptive same-mask error decomposition. No neural
training, model selection, threshold tuning or new evaluation role in this
control. All365 original stationary fit rows and source assumptions remain.

`cached_verified`: all18 original predictions replay exactly. A completed-run
resume verifies18 receipts with0 fresh evaluations. Measured control processing
time2.58s, not a training cost. Local row predictions remain outside Git.

## The Input-Support Hypothesis

Compare unchanged input, training-only camera-Jacobian box projection,
training-only projection of all32 geometry features, and unchanged input with
fallback outside that marginal box. Images, output coordinate H and labels are
never clipped. No held data defines support. Training rows remain unchanged.

The complete [table](results.md) and [metrics](metrics.json) preserve all settings.
For the past-RGB arm, three-seed guarded ADE gain versus CV is:

| Held scene | Original | Camera-only box | All-feature box | Outside-support fallback |
| --- | ---: | ---: | ---: | ---: |
| ETH | -14.98% | -15.50% | -16.15% | 0.00% |
| Hotel | -172.63% | -110.86% | -82.86% | 0.00% |

None of72 settings has positive guarded gain. The fallback column has exactly
zero intervention: all81ETH and284Hotel rows lie outside at least one training
feature interval. It is not cross-scene success or evidence that extrapolation
is inherently impossible. A high-dimensional marginal box with only one source
scene is a conservative and possibly uninformative support rule.

The input diagnosis finds97.53%of heldETH rows outside the Hotel training range
for the nearest-surface and two camera components. The reverse direction also
has substantial camera, geometry and neighbor mismatch. This differs from the
earlier +/-10 standardized clipping statistic; being outside observed minima/
maxima is a stricter criterion. Do not confuse the two counts.

## What Is Prediction and What Is Abstention?

The [post-fit decomposition](decomposition.json) fixes the original switch mask
while substituting the treated trajectory, then changes the gate. This is
path-dependent error accounting, not causal identification or a new policy.

For geometry-only Hotel, camera clipping appears to improve guarded gain from
-132.04% to -98.12%. But with the original switch mask held fixed, it is
-135.32%: the trajectory component worsens, and reduced intervention explains
the apparent gain. This rules out interpreting that number as dynamics lift.

For past-RGB Hotel, all-feature clipping does improve the same-mask prediction
component (-172.63% to -113.77%); the new gate then reduces harm to -82.86%.
Both remain worse than CV. ETH visual arms worsen even with their original
switch masks. Thus camera/context support contributes to some errors but does
not supply the missing generalizable direction prediction.

Easy CV error iszero on still rows, so percentage degradation is undefined;
absolute harm remains reported and positive for visual clipping arms. Native
errors accompany normalized ones. Large relative losses must not be mistaken
for verified physical distances: ETH/Hotel native CV ADE is approximately
0.24997/0.02117 in supplied coordinate units, not verified meters.

## Follow-Up, Not Promotion

This evidence motivated one registered matched retraining experiment, removing
only the four learned camera inputs while retaining deterministic output
conversion. Its [six-model result](../appearance_no_camera/conclusions.md) also
fails to beat CV. No clipped treatment, support rule or retrained seed is chosen
for deployment. Further threshold/feature-box sweeps on these exposed scenes
are not justified by these results.

The remaining priority is broader verified past-context training support and
useful forecasts, then independent utility/harm calibration. No new primary
metric, final-test claim, independent scene CI, physical safety, Stage5C or SMC.
