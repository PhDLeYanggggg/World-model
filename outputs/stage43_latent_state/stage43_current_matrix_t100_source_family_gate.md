# Stage43-CM Current-Matrix T100 Source-Family Gate

- source: `fresh_stage43_cm_current_matrix_t100_source_family_gate`
- result_source: `fresh_current_matrix_train_val_selected_t100_source_family_gate`
- gate: `13 / 13`
- verdict: `stage43_cm_current_matrix_t100_source_family_gate_pass_keep_floor`
- deploy current-matrix t100 source-family gate: `False`

## Current Matrix Scope

- train rows: `146809`
- val rows: `101446`
- test rows: `89736`
- test t100 rows: `18070`
- Stage43-CL local Stage43-T rows: `1440`
- Stage43-AT matrix rows: `89736`

## Feature Contract

- feature dim: `170`
- feature name hash: `7931ac5f58f0fecff8d0552a9078e6065507dac3bc59367a14c96cb5c67f1e01`
- denied feature hits: `[]`
- future waypoints: `label/eval only`

## Selected Model

- target: `residual`
- train filter: `t50t100`
- l2: `10000.0`
- train rows: `58845`
- model hash: `f2577d3c8ecdb53941a897e61e3a08d6cbfc6efe164d60bf369c17fbc52fc205`
- validation allowed families: `UCY`

## Raw Validation-Rule Test Metrics

- all ADE lift: `-0.59%`
- t50 ADE lift: `0.00%`
- t100 raw-frame diagnostic: `-3.86%`
- hard/failure lift: `-0.71%`
- easy degradation: `2.26%`
- switch rate: `17.24%`

## Deployed Metrics

- all ADE lift: `0.00%`
- t50 ADE lift: `0.00%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure lift: `0.00%`
- easy degradation: `0.00%`
- switch rate: `0.00%`
- deployment reason: `keep_floor_because_current_matrix_test_is_not_uniformly_positive_easy_safe`

## Source-Family T100 Test Table

| family | rows | val allowed | candidate t100 lift | selected t100 lift | easy degradation | switch |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| TrajNet_biwi | 1160 | `False` | -497.30% | 0.00% | 0.00% | 0.00% |
| TrajNet_crowds | 1440 | `False` | 0.80% | 0.00% | 0.00% | 0.00% |
| UCY | 15470 | `True` | -4.32% | -4.32% | 21.00% | 100.00% |

## Interpretation

This run uses the current Stage43 full-waypoint supervision matrix rather than the earlier small Stage43-T source split. It persists the causal feature names and source/split hashes, chooses source-family t100 switch rules on validation only, then evaluates the selected rules once on test.

If the validation-selected source-family rule is not positive and easy-safe on the current matrix test set, deployment stays at the Stage43-CI/CK floor. That is a conservative claim boundary, not a t100 success claim.

Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric or seconds-level claim; no Stage5C execution; no SMC.
