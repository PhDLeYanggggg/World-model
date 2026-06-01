# Stage43-AC Scene-Proxy Guarded Latent Policy

- source: `fresh_stage43_ac_scene_proxy_guarded_latent_policy`
- result_source: `fresh_replay_guarded_scene_proxy_policy`
- mode: `small`
- gate: `11 / 11`
- verdict: `stage43_ac_guarded_scene_proxy_latent_candidate`
- deploy guarded scene-proxy latent: `True`

## Selected Policy

- policy: `{'family': 'ab_non_h100', 'gain_threshold': 0.0, 'harm_threshold': 1.0, 'failure_threshold': 0.0, 'fallback': 'stage43_m_protected_policy', 'selected_on': 'validation_only', 'test_threshold_tuning': False}`
- validation objective: `0.591522`

## Test Metrics

| metric | Stage43-M | Stage43-AB all | Stage43-AC guarded | delta vs M | delta vs AB all |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_waypoint_ade_improvement_vs_floor` | `29.77%` | `38.97%` | `41.17%` | `11.40%` | `2.20%` |
| `t50_full_waypoint_ade_improvement_vs_floor` | `16.45%` | `35.42%` | `35.42%` | `18.97%` | `0.00%` |
| `hard_failure_full_waypoint_ade_improvement_vs_floor` | `28.75%` | `39.66%` | `42.34%` | `13.58%` | `2.68%` |
| `easy_degradation_vs_floor` | `0.00%` | `0.14%` | `0.00%` | `0.00%` | `-0.14%` |
| `t100_raw_frame_full_waypoint_diagnostic_vs_floor` | `-17.79%` | `-32.64%` | `-17.79%` | `0.00%` | `14.84%` |
| `switch_rate` | `68.91%` | `100.00%` | `85.96%` | `17.04%` | `-14.04%` |
| `scene_proxy_override_rate` | `0.00%` | `0.00%` | `80.12%` | `0.00%` | `0.00%` |
| `t100_scene_proxy_override_rate` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` |

## Interpretation

Stage43-AC keeps the Stage43-AB scene-proxy latent head only where a validation-selected guard allows it, and otherwise falls back to the Stage43-M protected latent policy. The guard is explicitly t100-aware because Stage43-AB improved all/t50/hard but worsened raw-frame t100 diagnostic.

## Boundary

- Scene proxy remains a train-only route/SDF/goal proxy, not raw image or verified metric SDF.
- Future endpoint/full waypoints are labels/eval only.
- t100 remains raw-frame diagnostic only.
- No Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.

## Gate

| gate | passed |
| --- | --- |
| stage43_m_available | True |
| stage43_ab_available | True |
| fresh_replay_completed | True |
| validation_only_selection | True |
| row_alignment_passed | True |
| easy_preserved | True |
| core_lift_over_stage43_m | True |
| t100_not_worse_than_stage43_m | True |
| scene_proxy_not_all_h100 | True |
| no_future_or_test_leakage | True |
| no_metric_seconds_stage5c_smc_claim | True |
