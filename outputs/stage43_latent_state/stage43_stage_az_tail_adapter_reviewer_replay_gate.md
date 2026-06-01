# Stage43-AZ Tail Adapter Reviewer Replay

- source: `fresh_stage43_az_tail_adapter_reviewer_replay`
- result_source: `fresh_exact_recompute_replay_from_stage43_p_artifact`
- verdict: `stage43_az_tail_adapter_reviewer_replay_pass`
- gate: `12 / 12`
- reviewer replay passed: `True`
- policy hash: `9155067aacf42bc8d8e67745c1cf5e05b729f95a88cf65d33d88b9a06c21484b`
- model hash match: `True`
- replay max metric diff: `0.0000000000`

## Replay Mode

- mode: `artifact_selected_config_and_allowed_rules_only_no_validation_reselection_no_test_threshold_tuning`
- artifact: `outputs/stage43_latent_state/stage43_tail_horizon_waypoint_adapter.json`
- artifact sha256: `3ff77bead78d1117650d4728573df4581a100f84a38c5728448bf0e2daf5a0b5`
- switch hash: `ae3d96350060581e492bc707400a4e19f780dee2e98e67cf671623e01263d55c`
- feature schema hash: `9144b865a6dc9903c427b75a33d44a3314ea4de4dcb2073cd1a353947c036733`

## Replayed Metrics

- all full-waypoint ADE improvement: `50.25%`
- endpoint FDE improvement: `51.15%`
- t50 full-waypoint ADE improvement: `51.23%`
- t50 endpoint FDE improvement: `55.13%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure full-waypoint ADE improvement: `47.88%`
- easy degradation: `0.00%`
- switch rate: `70.45%`

## Metric Replay Diff

| metric | artifact | replayed | abs diff |
| --- | ---: | ---: | ---: |
| `full_waypoint_ade_improvement_vs_floor` | `50.25%` | `50.25%` | `0.0000000000` |
| `endpoint_fde_improvement_vs_floor` | `51.15%` | `51.15%` | `0.0000000000` |
| `t50_full_waypoint_ade_improvement_vs_floor` | `51.23%` | `51.23%` | `0.0000000000` |
| `t50_endpoint_fde_improvement_vs_floor` | `55.13%` | `55.13%` | `0.0000000000` |
| `t100_raw_frame_full_waypoint_diagnostic_vs_floor` | `0.00%` | `0.00%` | `0.0000000000` |
| `hard_failure_full_waypoint_ade_improvement_vs_floor` | `47.88%` | `47.88%` | `0.0000000000` |
| `easy_degradation_vs_floor` | `0.00%` | `0.00%` | `0.0000000000` |
| `switch_rate` | `70.45%` | `70.45%` | `0.0000000000` |

## Boundary

- This is an exact recompute replay from the Stage43-P artifact; no validation reselection and no test threshold tuning.
- Dataset-local/raw-frame 2.5D only.
- t100 remains raw-frame diagnostic and guarded.
- No true 3D, foundation, metric/seconds, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `stage43_p_artifact_present` | `True` |
| `stage43_p_artifact_passed` | `True` |
| `model_hash_exact` | `True` |
| `feature_standardization_hashes_match` | `True` |
| `split_hashes_recorded` | `True` |
| `switch_hash_recorded` | `True` |
| `replay_metrics_exact` | `True` |
| `replayed_policy_safe` | `True` |
| `no_future_or_test_leakage` | `True` |
| `claim_boundary_not_overstated` | `True` |
| `stage5c_and_smc_false` | `True` |
| `long_objective_kept_active` | `True` |
