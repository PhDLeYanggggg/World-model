# Stage42-DN Deployment Variant Card

- source: `fresh_deployment_variant_card_from_stage42_cr_cq_di_dl_dm`
- generated_at_utc: `2026-05-26T23:23:16.195382+00:00`
- gate: `20 / 20`
- verdict: `stage42_dn_deployment_variant_card_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DN 是 deployment variant card，不重新训练，不调 threshold。
- DN 区分 safety-sensitive deployable、accuracy-priority diagnostic、source-level full-waypoint runtime policy，避免混用 claim。
- future waypoints / endpoints 只作为 supervised/evaluation labels，不能作为 inference input。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Deployment Variants

| variant | role | status | comparison baseline | all | t50 | t100 raw | hard | easy | safety note |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `endpoint_linear_reference` | `reference_floor` | `reference_only` | `endpoint_linear_bridge` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | near@0.05 0.0 |
| `no_proximity_guard` | `accuracy_priority_diagnostic` | `diagnostic_not_safety_sensitive` | `endpoint_linear_bridge` | `0.030166976195252437` | `0.01502943431774939` | `0.06118732156780071` | `0.03280214089079592` | `0.002532827637569346` | worsens near-collision@0.05 versus endpoint-linear |
| `proximity_guard` | `safety_sensitive_deployable_bridge_shape_policy` | `deployable_when_joint_proximity_safety_is_required` | `endpoint_linear_bridge` | `0.017743597342181783` | `0.010673426149055754` | `0.03480124336134305` | `0.01929354722729537` | `0.0024927762207753723` | near@0.05 -0.0006053046700172371 |
| `group_consistency_full_waypoint_runtime` | `source_level_full_waypoint_group_consistency_runtime_policy` | `runtime_ready_for_its_source_level_protocol` | `train_horizon_causal_floor_not_endpoint_linear_bridge` | `0.24715658317833844` | `0.2236298792899738` | `0.1434611214781808` | `0.23887420070464105` | `-0.2563085406508494` | base near 0.019364490707573012 -> final 0.01382274853554722 |

## Recommended Use

- safety_sensitive_default: `proximity_guard`
- accuracy_priority_diagnostic: `no_proximity_guard`
- source_level_full_waypoint_runtime_candidate: `group_consistency_full_waypoint_runtime`
- deployment_floor: `Stage37 / teacher floor remains required`

## Claim Boundary

- Do not present `no_proximity_guard` as safety-sensitive deployment.
- Do not rank-mix `group_consistency_full_waypoint_runtime` directly against endpoint-linear composer variants without stating its different train-horizon causal-floor comparison baseline.
- t+100 remains raw-frame diagnostic; dataset-local coordinates remain non-metric.
- Stage5C remains unexecuted and SMC remains disabled.

## Gate

| gate | pass |
| --- | --- |
| `cr_ablation_passed` | `True` |
| `cq_guard_passed` | `True` |
| `di_group_repair_passed` | `True` |
| `dl_runtime_passed` | `True` |
| `dm_reviewer_replay_passed` | `True` |
| `no_guard_marked_diagnostic` | `True` |
| `no_guard_proximity_caveat_visible` | `True` |
| `proximity_guard_marked_safety_deployable` | `True` |
| `proximity_guard_near_collision_not_worse` | `True` |
| `proximity_guard_ci_positive` | `True` |
| `group_runtime_marked_protocol_specific` | `True` |
| `group_runtime_exact_replay_visible` | `True` |
| `group_runtime_near_collision_reduced` | `True` |
| `baseline_mixing_caveat_present` | `True` |
| `recommended_policy_declared` | `True` |
| `no_metric_seconds_overclaim` | `True` |
| `true_3d_overclaim_blocked` | `True` |
| `foundation_overclaim_blocked` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |
