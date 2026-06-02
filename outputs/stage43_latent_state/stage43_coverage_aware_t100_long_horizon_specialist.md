# Stage43-CJ Coverage-Aware T100 Long-Horizon Specialist

- source: `fresh_stage43_cj_coverage_aware_t100_long_horizon_specialist`
- result_source: `fresh_stage43_cj_coverage_aware_t100_long_horizon_specialist`
- gate: `15 / 15`
- verdict: `stage43_cj_t100_long_horizon_specialist_pass_keep_ci_floor`
- deploy t100 specialist: `False`
- checkpoint committed: `False`

## Boundary

- This is a t100-only neural specialist trained from past/current causal features plus Stage43-CG latent outputs.
- Future waypoints are labels/evaluation targets only, never inference inputs.
- Dataset-local/raw-frame 2.5D only; no metric or seconds-level claim.
- Stage5C not executed; SMC not enabled.

## Deployed Test Metrics

- all full-waypoint ADE improvement: `52.03%`
- t50 full-waypoint ADE improvement: `31.13%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure improvement: `50.48%`
- easy degradation: `0.00%`
- switch rate: `69.09%`

## Trial Candidate Test Metrics

- candidate all full-waypoint ADE improvement: `52.03%`
- candidate t100 raw-frame diagnostic: `0.00%`
- candidate hard/failure improvement: `50.48%`
- candidate easy degradation: `0.00%`

## Delta vs Stage43-CI Floor

- all delta: `0.00%`
- t50 delta: `0.00%`
- t100 delta: `0.00%`
- hard/failure delta: `0.00%`
- easy degradation delta: `0.00%`

## Bootstrap CI

- bootstrap n: `2000`
- all CI: `[51.65%, 52.41%]`
- t50 CI: `[30.48%, 31.78%]`
- t100 CI: `[0.00%, 0.00%]`
- hard/failure CI: `[50.04%, 50.90%]`

## Horizon Table

| horizon | rows | ADE improvement | endpoint improvement | switch | easy degradation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 15385 | `71.83%` | `74.96%` | `97.71%` | `0.00%` |
| 25 | 13913 | `60.96%` | `68.43%` | `87.06%` | `0.00%` |
| 50 | 12259 | `31.13%` | `41.49%` | `60.38%` | `0.00%` |
| 100 | 8443 | `0.00%` | `0.00%` | `0.00%` | `0.00%` |

## Interpretation

selected t100 specialist did not pass positive/easy-safe test gate; keep Stage43-CI floor

## Gate

| gate | passed |
| --- | --- |
| `ci_precondition_passed` | `True` |
| `fresh_torch_training` | `True` |
| `checkpoint_not_committed` | `True` |
| `validation_selected` | `True` |
| `no_test_threshold_tuning` | `True` |
| `future_waypoints_label_only` | `True` |
| `no_future_endpoint_or_central_velocity` | `True` |
| `no_test_goal_or_stat_leakage` | `True` |
| `all_still_positive` | `True` |
| `t50_not_destroyed` | `True` |
| `hard_failure_still_positive` | `True` |
| `easy_preserved` | `True` |
| `t100_result_honest` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
