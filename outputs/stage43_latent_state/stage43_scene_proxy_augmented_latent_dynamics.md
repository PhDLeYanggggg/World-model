# Stage43-AB Scene-Proxy Augmented Latent Dynamics

- source: `fresh_stage43_ab_scene_proxy_augmented_latent_dynamics`
- result_source: `fresh_run_scene_proxy_augmented_torch_training`
- mode: `small`
- gate: `11 / 11`
- verdict: `stage43_ab_scene_proxy_augmented_latent_lift_candidate`
- deploy scene-proxy augmented neural: `True`
- checkpoint committed: `False`

## Feature Integration

- base feature count: `162`
- scene proxy feature count: `14`
- total feature count: `176`
- scene proxy hashes: `{'train': '9f2b890a668b26884eba095014be24f793a4bdbe6bed52e0cfeb4d76f7634887', 'val': 'ed14f45e3a7b642d892d46df4e014b18f52ca3b9d052dd1a5ff9c3f62e929600', 'test': '0f5ac09955b9799cd878198f46f0116a8ebcc18cf29a03f09c3bc1fbc35a23b7'}`

## Metrics vs Floor

| metric | Stage43-M | Stage43-AB | delta |
| --- | ---: | ---: | ---: |
| `full_waypoint_ade_improvement_vs_floor` | `29.77%` | `38.97%` | `9.20%` |
| `t50_full_waypoint_ade_improvement_vs_floor` | `16.45%` | `35.42%` | `18.97%` |
| `hard_failure_full_waypoint_ade_improvement_vs_floor` | `28.75%` | `39.66%` | `10.91%` |
| `easy_degradation_vs_floor` | `0.00%` | `0.14%` | `0.14%` |
| `t100_raw_frame_full_waypoint_diagnostic_vs_floor` | `-17.79%` | `-32.64%` | `-14.84%` |
| `switch_rate` | `68.91%` | `100.00%` | `31.09%` |

## Interpretation

Scene/raster proxy features improved at least one Stage43-M comparison slice and remain protected by the same floor. This is a candidate for promotion, but still dataset-local/raw-frame 2.5D and proxy-only.

## Boundary

- Scene proxy is train-only route/SDF/goal prior, not raw imagery and not verified metric SDF.
- Future endpoint/full waypoints are labels/eval only.
- No Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.

## Gate

| gate | passed |
| --- | --- |
| stage43_m_baseline_available | True |
| stage43_aa_precondition_passed | True |
| torch_training_fresh_run | True |
| checkpoint_not_committed | True |
| scene_proxy_features_integrated | True |
| scene_proxy_hashes_recorded | True |
| protected_eval_completed | True |
| easy_preserved | True |
| scene_proxy_lift_or_honest_not_promoted | True |
| no_future_or_test_leakage | True |
| no_metric_seconds_stage5c_smc_claim | True |
