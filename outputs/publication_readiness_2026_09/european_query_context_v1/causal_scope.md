# Causal Scope and Reused Lineage

This records code/lineage verification, not an independent statistical safety
certificate or a new dataset audit. The numerical packet hashes and preprocessing
are checked again against the frozen anchor before fitting.

## Input Path
`m3w_floor_relative.matched_features` builds380columns from causal geometry,
the fixed reference, neural forecast and old causal floor decision.
`m3w_native_gain_harm.cost_features` and
`m3w_supervised_intervention.risk_features` use history, history masks, up to
eight neighbor histories/masks and forecast diagnostics. Predicted endpoints
are allowed; ground-truth future endpoints and future-valid masks are absent.
No scene goals are rebuilt, and no central velocity is introduced.

`run_m3w_inner_separability.training_arrays` retrieves rows by source identity,
not future outcome. Upstream query sampling uses only site/recording/frame hashes.
`run_m3w_temporal_auxiliary.inputs` exports only the original TRAIN recordings.
The current `m3w_shared_train_risk.prepare` applies the unchanged exhaustive,
disjoint whole-recording inner-fit/held split. The forecasting producer and
original controller exclude all four target source localities in each context.
The subsequent cost heads do learn on those localities' inner-fit recordings;
therefore the new readout is not unseen-locality generalization.

## New Context Operation
The only grouping keys are the current recording and frame within a single
source packet. All available feature rows participate, including rows lacking
future labels and rows not currently eligible for intervention. Pooling never
receives labels, label-valid masks, outcomes, action masks or neighboring frames.
Single-row queries use their own input. Preprocessing is frozen from inner-fit
labels/features, as in the existing supervised comparison; no held statistics
are estimated. The feature means are pooled only within the current query,
not over an entire held recording or held dataset.

The cached upstream neighbor cap and focal-row admission rules remain unchanged.
This context does not recover undetected agents, add image semantics, identify
physical interactions or create a new scene pack. All source features are
ego-relative before their existing standardization; pooling them supplies a
query summary, not a common-world-coordinate collision model.

## Verification
Tests cover leave-self-out arithmetic, singleton behavior, row-permutation
equivariance, exclusion of different frames/recordings, retention of an unknown
label peer, pooling before inference batching, exact resume and frozen reference
outputs. Full real readout additionally replays all eight old prediction hashes
and policy summaries, then computes matched counts within each query.

Independent selection/calibration/confirmation and outer development stay
closed. Historical exposed scores remain exploratory. The2% selected positive
easy-harm criterion is unchanged and differs from ordinary net easy ADE change.
Missing outcomes are not declared safe. Stage5C and SMC remain disabled.
