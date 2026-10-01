# Component Calibration Removes Coverage, Not the Remaining Risk Failure

This is an exposed-development diagnostic on frozen upstream43 assets and three
cost-head seeds. It is not an independent generalization test, new neural dynamics
training, or a deployment result. Source-only calibration and frozen transfer
evaluation are fresh_run; the underlying72 heads are cached_verified.

## Hypothesis Tested

The crossed-head study exposed both harm underestimation and reference
overestimation. This experiment fitted empirical recording-level corrections
for those components, with whole-recording cross-fitting and fixed arms.
No transfer outcomes chose a margin, quantile, arm or seed. The2% selected-reference
budget, prediction task and independent-role restrictions did not change.

## What Changed and What Did Not

Parent source-screened easy-risk failures decline from18/90 defined directions
to6/60 under joint calibration. Undefined directions grow from126 to156, and
ADE gain against the same floor declines from0.047605% to0.034200%. At the same
per-query intervention counts the parent also fails6/60, and its ADE gain is
0.037282%, above the calibrated policy. Fewer failing directions therefore do
not establish better selection or a safe policy. Joint risk remains above2%,
with worst ratio2.278217%.

Reference-only screened ADE gain is0.034204%, almost identical to joint0.034200%.
It has the same6/60 risk-failure count. In this fixed control, adding the harm
adjustment does not establish an incremental predictive contribution. This
does not prove that harm estimation never matters: the previous decomposition
identified genuine underestimation in some directions.

## A Concrete Nonmonotonicity Counterexample

For source008 -> target020 in single1/controller0, head43 has parent easy
selected risk1.649600%, inside the2% budget. Joint calibration reduces the
intervention rate from0.390016% to0.234009%, but observed selected risk rises
to2.104243%, outside the budget. Count-matching the parent produces the same
risk here. These are already-exposed development outcomes, not a new test.

This is compatible with conservative predicted component changes. Removing
selected rows can remove low-harm reference mass faster than positive harm.
The actual harm/reference ratio is not generally monotone in the size of the
selected set. Tightening predicted harm/reference components is not, by itself,
a proof of actual selected-tail safety. The actual future outcomes cannot be
used to decide which rows to retain.

## Remaining Failing Directions

| Context; source -> target; head | Joint easy risk |
|---|---:|
| single1/controller0;008 ->020;17 |2.104243%|
| single1/controller0;008 ->020;43 |2.104243%|
| single1/controller0;048 ->067;17 |2.278217%|
| single1/controller0;048 ->067;43 |2.147762%|
| single1/controller2;126 ->112;29 |2.047815%|
| single1/controller2;126 ->112;43 |2.117262%|

These six views repeat sources/targets and are not six independent scenes.
Some have positive whole-easy ADE improvement while selected positive-harm
risk fails: the metrics measure different properties and must not be exchanged.

## Support and Estimator Limitations

There are348 leave-one-recording-out folds but only58 distinct source validation
recordings.29/72 full calibrators have no known raw-eligible source rows; none
of those29 has an unknown-only eligible pool. This is not missing data files.
Among full calibrators,11 have only one contributing recording for their least
supported component,17 have two, and121 OOF folds lack component support.
Computing a coefficient from one recording is not a reliable uncertainty claim.

The margins summarize the old raw-eligible pool. After calibration, the selected
subset changes. Its conditional errors may differ from the pool used to fit
the margins. In addition, source-only recording errors may not transport to a
different locality. This experiment cannot separate those two mechanisms from
source statistics alone, nor can resubstitution success validate either one.

The method does not fit conditional quantile regressors, does not implement
CQR and does not prove exchangeability across scenes. It has no conformal,
population-risk or physical-safety guarantee. See [reading note](related_work_note.md).

## Next Action, Not a Claimed Repair

Keep the floor and all negative controls. Before another margin or threshold
trial, check existing selected-subset and scene-joint assets, then register an
OOF selected-pool accounting diagnostic. Track benefit, positive harm and
reference mass retained/removed by each frozen policy on source OOF recordings
and transferred localities. Separate within-source post-selection change from
cross-locality calibration drift. Do not refit on the six transferred failures.

If the within-source subset change dominates, the candidate repair should learn
or calibrate the final policy's selected-set risk, with a recording-disjoint
evaluation of that policy, rather than reuse raw-pool residual margins. If source
OOF support itself is insufficient, report that gap and address causal feature
or data support before opening independent confirmation. Next experiment not_run.

Obs8/pred12, stride12 raw frames, image-local detector-silver only. No metric,
seconds, true3D, foundation, human-gold or physical-safety claim. Stage5C execution
and SMC remain off; no deployment promotion follows from this run.
