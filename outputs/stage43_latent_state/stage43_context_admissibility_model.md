# Stage43-BT Context Admissibility Model

- source: `fresh_stage43_bt_context_admissibility_model`
- result_source: `fresh_row_level_harm_aware_context_admissibility`
- verdict: `stage43_bt_context_admissibility_pass_safe_lift_diagnostic`
- gate: `14 / 14`
- row-level admissibility trained: `True`
- beats graph-history on any core metric: `True`
- easy safe: `True`
- deployable policy changed: `False`

## Validation-Selected Policy

- selected policy: `{'gain_threshold': 0.5, 'harm_threshold': 0.5, 'predicted_gain_threshold': 0.0}`
- candidate count: `125`
- safe candidate count: `125`
- context counts on validation: `{'graph_history_only': 9543, 'scene_graph_full': 712, 'scene_proxy_only': 1745}`

## Test Metrics

- all full-waypoint ADE improvement: `39.06%`
- t50 full-waypoint ADE improvement: `16.02%`
- t100 raw-frame diagnostic improvement: `-3.21%`
- hard/failure improvement: `39.66%`
- easy degradation: `0.00%`
- switch rate: `57.44%`
- context counts on test: `{'graph_history_only': 10322, 'scene_graph_full': 531, 'scene_proxy_only': 1147}`

## Delta Vs Graph-History-Only

- all delta: `2.15%`
- t50 delta: `0.40%`
- hard/failure delta: `2.02%`
- easy degradation delta: `0.00%`

## Admissibility Diagnostics

| variant | mean true gain | mean predicted gain | gain-label rate | harm-label rate | gain correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scene_proxy_only` | `-0.00634` | `-0.02337` | `26.82%` | `28.82%` | `0.2327` |
| `scene_graph_full` | `-0.01021` | `-0.01428` | `23.86%` | `36.46%` | `0.0962` |

## Interpretation

- This is a row-level harm-aware context admissibility diagnostic.
- It tries to release scene/full context only when predicted gain is high and predicted harm is low.
- Thresholds are selected on validation only; test is evaluated once.
- Future variant errors are labels/eval only, not inference inputs.
- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `bp_precondition_passed` | `True` |
| `bq_precondition_passed` | `True` |
| `br_precondition_passed` | `True` |
| `bs_precondition_passed` | `True` |
| `fresh_torch_training_completed` | `True` |
| `checkpoint_not_committed` | `True` |
| `train_val_test_loaded` | `True` |
| `validation_only_threshold_selection` | `True` |
| `test_eval_completed` | `True` |
| `graph_history_reference_present` | `True` |
| `admissibility_diagnostics_reported` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
