# Dedicated-Role Calibration: Working Paper Addendum

This result note supplements the frozen v8 working manuscript. The v8 main
PDF has not been regenerated to include this experiment. This is not a
submission-ready paper, independent final confirmation or a deployment claim.

## Question And Frozen Design

Can a single risk-only adjustment make an already trained cost family meet the
original recording-level error-increase requirements without changing its
utility ranking? The experiment uses the previously reserved calibration role:
102 recordings in 12 locality groups, with at most 256 query frames per
recording. All groups are retained. There are 481,537 past-supported targets;
one locality supplies 86.35% of those overlapping target windows.

Three SOURCE forecaster folds and three actual training seeds produce nine
frozen pipelines. Cost models and normalization are SOURCE-only. No optimizer
updates occur during this experiment. The floor is fixed by each SOURCE
pipeline, not chosen using calibration outcomes. All four allocation arms and
all 13 multipliers freeze for every recording before future targets are built.

For native predicted benefit B, positive harm H, reference R, easy reference ER
and easy harm EH, utility remains B-H. Lambda tightens only the predicted
all/easy risk constraints: lambda H <= 0.02 R and lambda EH <= 0.02 ER.
The coefficient implementation divides R and ER by lambda; this transformed
optimizer array is not a new coherent cost posterior. Matched joint allocation
retains the independent arm's intervention count at every query. Free-count
allocation does not have that matching. These are shared error-budget
constraints, not physical pairwise collision constraints.

The registered choice is the least common multiplier passing every original
recording/pipeline view for both independent and matched joint policies.
Undefined support does not pass. No new model, seed, scene or utility criterion
is selected from calibration outcomes.

## Complete Readout

All 918 recording/pipeline grids completed in 12,106.17 seconds locally. The
subsequent target construction and scoring retain the full cohort. The raw
future CSV rows may have been parsed during input conversion; the sequencing
claim concerns supervised target windows and predictive error readout. Input
features are restricted to each query's present and past, separately verified
by prefix reconstruction. No eventual label mask controls the decision.

There is **no supported common multiplier** under the registered contract.
The following fixed endpoint illustrates the utility/safety difference; it is
not a newly chosen operating point.

|Lambda 1 policy|Available-label ADE gain % [95% CI]|Completion-lower gain % [95% CI]|Supported views / 918|Worst known easy net %|
|---|---:|---:|---:|---:|
|Independent|1.4121 [1.0922, 1.7757]|0.8393 [0.3794, 1.2931]|6|41.0795|
|Joint, matched count|1.9255 [1.4393, 2.4855]|1.0048 [0.4593, 1.5714]|0|103.5715|

Utility first pools native reference mass within each locality and then weights
the 12 locality ratios equally. Intervals use 3,000 locality resamples and are
conditional calibration summaries, not a general population risk certificate.
Repeated models and overlapping windows do not create independent localities.

The direct matched-minus-independent contrast at lambda 1 is +0.5134 pp
[0.2701, 0.8014] on known outcomes, but only +0.0697 pp [-0.0946, 0.2628] for
the paired completion-lower effect. This latter calculation uses the actual
policy-pair disagreement on wholly unknown rows, not a subtraction of two
individual lower bounds. Some other fixed grid contrasts have positive lower
intervals; all are retained in the full report. None replaces the unsupported
registered selection or supplies multiplicity-controlled superiority.

## Failure Attribution

The pre-readout taxonomy separates observed error harm, unknown-row bounds and
unavailable selected reference. At lambda 1, the matched joint policy has 907
exclusive observed-harm failures, 9 unsupported views and 2 unknown-bound-only
failures. The overlapping observed-harm count is 909. This is not primarily a
case in which all outcomes are missing. Across unique target rows, 471,499
have some future labels, 281,603 have all 12, 189,896 are partial, and 10,038
are wholly unknown. No full-roster known-easy reference is absent in any of
the 918 views; missing selected support is a distinct issue.

A post-readout, explicitly descriptive attribution reconciles selected cost
masses with all 23,868 original metric views for the two required arms, each
with two known-risk ratios (47,736 scalar comparisons).
At lambda 1, matched-joint record medians of predicted/observed cost are:

|Native cost|Median predicted / observed|Defined recording/pipeline views|
|---|---:|---:|
|H|0.195606|912|
|EH|0.0000310265|906|
|R|1.048541|915|
|ER|0.013008|909|

These are conditional medians of defined ratios, not ratios of pooled costs
or independent samples. Ordinary and easy harm are severely underestimated in
the selected region. Easy reference is also underestimated; a generic claim
that inflated easy reference alone explains the result would be incorrect.
The exact excess decomposition separates these contributions, but does not
prove whether the underlying training cause is rare-event loss, saturation,
representation, scene shift or another mechanism.

Tightening lambda reduces interventions: independent allocation falls from
29.65% to 0.137% across the full grid. It does not fix the missing tradeoff.
At lambda 4096, matched joint known gain is only 0.00210%, with 141 supported,
675 unsupported, 96 exclusive observed-harm and 6 unknown-bound-only views.
The worst signed easy net is below 2%, but separate positive-harm/reference
checks still fail. Average preservation and signed net are not those checks.

## Scientific Consequence And Next Experiment

The frozen score family is not repaired by the registered scalar calibration.
No new learned controller is deployed; the global floor is a distinct fallback,
not a successful learned policy. The narrower evidence is that shared-budget
allocation can improve descriptive utility while still failing original
recording-level risk. This does not establish a new world-model capability.

Before another training change, compare the same mean_absolute critics with
their matched original controls on producer-excluded SOURCE data. Measure the
same H/EH/R/ER quantities in all-legal and selected regions. SOURCE fitting
readout is optimistic, not independent validation. The decisive distinction is
whether the severe easy-harm error is already present on fitting data or is
principally amplified on new localities. Check earlier failed easy-harm,
conditional-moment and boundary-loss experiments before repeating a loss arm.
Any subsequent architecture or threshold informed by the current outcomes must
disclose this cohort's exposure. Final confirmation and DroneCrowd stay closed.

## Claim Boundaries

The task remains obs8/pred12 at raw stride12, local image pixels and released
detector/tracker silver, not human gold. Available-label ADE is not full12 ADE.
The bounds cover wholly unknown rows only; they do not recover missing steps
within partial trajectories. Missing endpoint FDE is diagnostic. Past-only
released-series conditioning is not proof of causal online tracking. There is
no metric, seconds, physical-safety, true3D or foundation-model claim. Stage5C
and SMC remain off. Strong public-method comparisons, independent confirmation
and a full cold-start model reproduction package remain research deliverables,
not accomplishments implied by these reports or by successful program exit.
