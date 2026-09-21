# Native Gain/Harm Head Comparison

## Material Passport

- Date: 2026-09-21; prospective registration before new head fits or readout.
- Scope: source-only exploratory conditional-cost learning.
- Parent: verified native_nested_v1 physical training views and outer forecasts.
- No independent calibration, confirmation, threshold search or deployment.

## Question and Fixed Comparison

The native-loss forecast improves mean source ADE but harms some easy queries.
Can costs learned from clean nested producers identify useful interventions,
and does asymmetric harm fitting repair conditional harm underprediction?
This is a falsifiable learning comparison, not a new-architecture novelty claim.

Keep the four admitted SDD training sites, all 175,756 indexed source queries,
three seeds17/29/43, observation8/prediction12 and frozen native forecasts.
All four sites remain design-exposed. The source fitting exclusions are clean;
the source research choices are not independent confirmation. Original val/test,
main/external and bookstore are closed. Do not fit or select from those roles.

For each of12outer-site/seed views fit:

1. Weighted native benefit/harm ridge, fixed alpha0.01 on a mean-loss normal equation.
2. Width64 GELU/Softplus neural benefit/harm regression with squared loss.
3. The identical neural head with fourfold squared penalty when harm is underestimated.

Also report the training scene-balanced constant, signed ridge and clipped ridge
predictions. These are controls, not extra selected winners. The asymmetric harm
output targets an upper expectile, not an expected mean, confidence bound or
calibrated failure probability. Raw negative ridge harm stays visible.

## Inputs and Supervision

Use the existing306past/rollout summary features, full12-step causal baseline
and candidate paths (48features), and log observed native-coordinate scale (1).
Total355. Scale is computed from history, not future displacement. No scene ID,
future position/support, oracle label or future goal is an inference feature.
Identical candidate/baseline paths have exactly zero benefit/harm by construction;
all heads enforce that known identity at prediction time, without future labels.

Targets are native ADE benefit=max(CV-neural,0) and harm=max(neural-CV,0), from
the fixed two-site-excluded producers. Each training row's producer excludes its
own scene and the head's outer scene, including learned preprocessing. Unknown
cost rows remain indexed and unknown; they do not enter supervised fitting or
normalizer estimates. Every supported training scene receives equal weight.
Feature mean/std, scalar mean-CV target scale and constant control are fitted on
supported training rows only. Native costs are recovered before evaluation.

The two neural losses share initialization, exact scene-uniform batches, optimizer
and fixed budget:3,000updates,batch256,AdamW lr0.001/weight_decay0.0001,clip5.
Checkpoint every500updates; heartbeat every100.24neural fits total72,000updates,
plus12ridge fits. No early stopping, held-source model/seed selection or refit
after outcomes. A100-update coupa/seed17/MSE pilot counts toward its final budget.
Use native arm64CPU4/interop1/workers0; real pilot measures local cost before the
fixed matrix. No CREATE job is necessary unless measurements establish need.

## Fixed Readout, Not a Deployment Search

Complete all36head endpoints before scoring outer source rows. Reuse the bound
single-held outer forecasts without retraining. Compare every head under both
fixed rules: predicted benefit>harm; and that rule plus harm<=0.1*benefit.
These are diagnostic decision slices, not validated safety thresholds. No rule
is selected for deployment. An unchanged forecast is not counted as intervention.

Report benefit/harm MSE, gain MAE, negative harm scores and conditional realized
gain/harm in each slice. Report harm-score deciles as descriptive calibration
checks, not fitted probability calibration. Compare actual ADE/FDE, complete-future
sensitivity, intervention rate, unknown selected outcomes, worst site and tails.
Primary summary remains equal mean of within-site native ADE improvement overCV;
three seeds and3,000physical-scene bootstrap resamples are conditional on four
explored sites, not thousands of independent windows or a generalization guarantee.

Report complete-future zero-CV queries separately: absolute added error must be
zero for the strict empirical protection check; percentage is undefined. Also
show positive-easy costs below the training-only positive-CV q25 and hard costs
above training-only CV q75. These additional native slices are diagnostic, not
a silently redefined formal easy population or a calibrated deployment gate.
No future-defined slice membership is used for inference decisions. Unknown
selected outcomes are counted explicitly, never assigned zero harm.

## Verification and Limits

Bind physical input/target archives, their manifests, all upstream checkpoints,
source assets and this code/config. Test future-time rejection, outer exclusion,
unknown label handling, exact interrupted resume and asymmetric loss gradients.
Verify all head endpoints, paired draw counts, train-only normalization and every
saved cost prediction on replay. Retain all results including failed safety.

Clean fit lineage does not remove prior design exposure or the two-training-site
versus three-training-site producer distribution shift. No new data, independent
calibration, multimodal contribution or safety guarantee is supplied by this run.
If costs remain unreliable, diagnose the largest conditional errors before a
new registered change. Do not rescue this experiment by choosing its best held
threshold, deleting zero-CV rows, or introducing a post-hoc pixel tolerance.
SDD pixel/raw-frame only; Stage5C and SMC remain disabled.
