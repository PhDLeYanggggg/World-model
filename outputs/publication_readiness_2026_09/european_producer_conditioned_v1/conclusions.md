# Producer Identity Does Not Yet Give A Robust Repair

I completed the registered experiment: 108 newly trained Torch gain/harm heads,
216,000 updates and all180 policy views. Training, decisions and evaluation are
fresh_run. Trajectory forecasts, source data and earlier reference models are
cached_verified. No trajectory forecaster was retrained or deployment changed.

## What Improved

The producer-tagged controller improves average ADE over its own two-source
branch floor in all36 dependent views, by 0.03472% to 2.40571%; all36 conditional
source-bootstrap intervals are positive. This is genuine observed controller
benefit on the opened development population, but the branch floor is not the
stronger, unchanged four-source system. The improvement cannot be transferred
to a different denominator or called independent confirmation.

The training path is reproducible: all108 heads reached2,000 updates, all216
checkpoint-prefix replays match, and unknown-label supervised draws are zero.
Fixed training-batch risk loss decreases for16/18 heads in each arm. Training
loss is not a validation or deployment-success metric.

## What Failed

On identical forecasts, the actual producer tag is not a stable improvement:

| Comparison | All-ADE gain range | Positive / negative conditional CI |
|---|---:|---:|
| Producer versus global | -0.50256% to +0.93933% | 13 / 9 |
| Producer versus placebo | -0.32664% to +1.02026% | 10 / 9 |
| Producer versus wrong-tag diagnostic | -0.85852% to +1.52248% | 12 / 9 |
| Producer versus unchanged four-source stop controller | -1.03775% to +2.27628% | 15 / 10 |

The exact values and all branch/event/seed breakdowns are in summary_metrics.json.
These interval counts describe correlated development views, not independent
discoveries or multiplicity-adjusted wins. Correct tag use is confounded with
the fitting-source cohort, and wrong tags sometimes perform better.

The producer policy's worst positive-easy locality degradation versus CV is
8.02939%, above the2% requirement. Its worst hard-locality degradation versus
the unchanged stop policy is15.12363%. Choosing only favorable branches would
hide these failures. The new policy is not promoted.

## Where The Easy Failure Comes From

Post-readout arithmetic retains a common CV denominator and separates the
two-source floor from the neural increment. There are12 already-bad floor
locality/views. Seven remain above2% under every tested policy; no controller
creates a new threshold crossing where its branch floor was below2%. Six of
the seven producer-policy violations persist despite a beneficial neural change.
All seven concern eu-locality-020 across different fold/seed/event views, not
seven independent localities.

In the worst producer case, branch-floor degradation is10.04054%; the neural
controller improves it by2.01115 percentage points, leaving8.02939%. Thus the
floor is an actual bottleneck. This does not excuse the final system's failure
or prove that a different floor would automatically solve transfer.

The learned risk budget also fails to certify realized harm. Across288 dependent
selected locality/views,102 have realized positive-harm ratio above2%, and233
underpredict this ratio. Realized ratio reaches15.25057%, while predicted ratios
range0.10587% to1.27096%. These are moment ratios, not net easy degradation.

## Decision And Next Experiment

Keep the existing deployment unchanged. A producer tag and a weaker two-source
fallback are not the repair. Do not search the opened readout for a better tag,
branch, threshold or seed.

Next I will test a source-disjoint three-role development construction that
keeps the same four-source producer/floor during controller fitting and readout:
producer sources A, cost-head sources B, and readout sources C. Rotate these
roles only within the twelve already-opened sources, preregister both directions,
and compare producer-matched supervision with the existing source-excluded
control on identical final forecasts. This proposal is not_run; it does not open
the reserved selection, calibration or confirmation roles. It also does not by
itself remove source-cohort confounding or guarantee risk calibration.

## Evidence Limits

The 180 choices,216 independent coordinate arrays,3,402 metric reductions and90
old reference metrics verify;344 tests pass in56 scoped files. The full legacy
suite was not_run. All decisions froze at16:02:43Z, before evaluation began at
16:02:54Z. The full training phase took547 seconds and evaluation445 seconds on
native arm64 CPU4/inter-op1/workers0. No resource failure or budget reduction.

Only four unique zero-CV examples support the no-observed-harm result: one
locality, two future labels per example, no endpoint. Unknown-label predictions
are not accuracy evidence. The data are released detector tracks, image-pixel
obs8/pred12 at raw annotation stride12, not historical raw-t50, seconds, metric,
human gold, physical safety, true3D or foundation evidence. Not submission-ready;
no independent confirmation, Stage5C execution or SMC.
