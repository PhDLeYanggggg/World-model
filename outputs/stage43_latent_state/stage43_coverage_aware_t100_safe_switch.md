# Stage43-CI Coverage-Aware T100 Safe Switch

- source: `fresh_stage43_ci_coverage_aware_t100_safe_switch`
- result_source: `fresh_stage43_ci_coverage_aware_t100_safe_switch`
- gate: `15 / 15`
- verdict: `stage43_ci_t100_safe_switch_pass_floor_repair`
- deploy t100 latent switch: `False`
- deploy t100 safe floor repair: `True`

## Boundary

- This repairs unsafe t100 switching; it does not claim t100 positive transfer unless the t100 metric is actually positive.
- Dataset-local/raw-frame 2.5D only.
- No metric or seconds-level claim.
- Stage5C not executed; SMC not enabled.

## Test Metrics After T100 Safe Switch

- all full-waypoint ADE improvement: `52.03%`
- t50 full-waypoint ADE improvement: `31.13%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure improvement: `50.48%`
- easy degradation: `0.00%`
- switch rate: `69.09%`

## Delta vs Stage43-CG Base Policy

- all delta: `0.56%`
- t50 delta: `0.00%`
- t100 delta: `5.51%`
- hard/failure delta: `0.76%`
- easy degradation delta: `0.00%`
- base t100 before repair: `-5.51%`

## Bootstrap CI

- bootstrap n: `2000`
- all CI: `[51.65%, 52.40%]`
- t50 CI: `[30.50%, 31.79%]`
- t100 CI: `[0.00%, 0.00%]`
- hard/failure CI: `[50.06%, 50.91%]`

## Horizon Table

| horizon | rows | ADE improvement | endpoint improvement | switch | easy degradation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 15385 | `71.83%` | `74.96%` | `97.71%` | `0.00%` |
| 25 | 13913 | `60.96%` | `68.43%` | `87.06%` | `0.00%` |
| 50 | 12259 | `31.13%` | `41.49%` | `60.38%` | `0.00%` |
| 100 | 8443 | `0.00%` | `0.00%` | `0.00%` | `0.00%` |

## Interpretation

The validation-selected policy repaired t100 by disabling unsafe t100 latent switching and falling back to the CE floor for t100 rows.

## Gate

| gate | passed |
| --- | --- |
| `cg_medium_precondition_present` | `True` |
| `ch_t100_blocker_confirmed` | `True` |
| `fresh_validation_selected_safe_switch` | `True` |
| `no_test_threshold_tuning` | `True` |
| `future_waypoints_label_only` | `True` |
| `no_future_endpoint_or_central_velocity` | `True` |
| `no_test_goal_or_stat_leakage` | `True` |
| `t50_not_destroyed` | `True` |
| `hard_failure_still_positive` | `True` |
| `all_still_positive` | `True` |
| `easy_preserved` | `True` |
| `t100_negative_repaired_to_nonnegative` | `True` |
| `t100_result_reported_honestly` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
