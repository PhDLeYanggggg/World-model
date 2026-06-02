# Stage43-BS Scene-Graph Context Router

- source: `fresh_stage43_bs_scene_graph_context_router`
- result_source: `fresh_validation_selected_scene_graph_context_router`
- verdict: `stage43_bs_scene_graph_context_router_pass_safe_no_lift_diagnostic`
- gate: `12 / 12`
- validation safe candidates: `5`
- selected route count: `0`
- unsafe full context blocked by BP prior: `True`
- deployable policy changed: `False`

## Selected Validation Router

- candidate: `{'min_gain': 0.01, 'min_rows': 100, 'allow_full': False}`
- selection rule: validation-only source/domain/horizon route table; fallback graph_history_only; test evaluated once
- val route variant counts: `{'graph_history_only': 12000}`
- test route variant counts: `{'graph_history_only': 12000}`

## Test Metrics

- all full-waypoint ADE improvement: `36.91%`
- t50 full-waypoint ADE improvement: `15.62%`
- t100 raw-frame diagnostic improvement: `-3.26%`
- hard/failure improvement: `37.64%`
- easy degradation: `0.00%`
- switch rate: `58.51%`

## Delta Vs Graph-History-Only

- all delta: `0.00%`
- t50 delta: `0.00%`
- hard/failure delta: `0.00%`
- easy degradation delta: `0.00%`

## Routes

| route | variant |
| --- | --- |

## Interpretation

- This router converts BR slice evidence into a validation-selected source/domain/horizon route table.
- It is a diagnostic context-routing experiment, not a deployment policy update.
- Route selection uses validation only; test is evaluated once.
- Future waypoints remain labels/eval only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `bp_precondition_passed` | `True` |
| `bq_precondition_passed` | `True` |
| `br_precondition_passed` | `True` |
| `validation_candidates_evaluated` | `True` |
| `validation_only_route_selection` | `True` |
| `route_table_nonempty_or_fallback_explicit` | `True` |
| `test_eval_completed` | `True` |
| `graph_history_reference_present` | `True` |
| `easy_safety_measured` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
