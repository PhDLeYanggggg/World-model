# Stage42-DE Full-Waypoint Deployment Gap Audit

- source: `fresh_stage42_de_full_waypoint_deployment_gap_audit`
- generated_at_utc: `2026-05-26T21:35:53.724693+00:00`
- git_commit: `ca67893`
- gate: `17 / 17`
- verdict: `stage42_de_full_waypoint_deployment_gap_audit_pass_primary_promotion_blocked`
- deployment_decision: `protected_full_waypoint_composer_supported_deployment_promotion_blocked`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DE 是 full-waypoint deployment-gap audit，不重新训练，不用 test 调 threshold，不执行 Stage5C，不启用 SMC。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- endpoint-only 或 endpoint-to-linear bridge 成功不能自动算 full-waypoint world-state dynamics 成功。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Decision Summary

- promote_full_waypoint_as_primary_deployable_dynamics: `False`
- use_guarded_full_waypoint_composer_for_safety_sensitive_reporting: `True`
- keep_endpoint_linear_or_stage37_teacher_floor_as_safety_floor: `True`
- blockers: `['protected_full_waypoint_does_not_beat_endpoint_linear_on_all_and_hard', 'ungated_full_waypoint_easy_degradation_unsafe', 'source_legal_time_t100_closure_open', 'graph_group_interaction_has_proximity_caveat']`

## Support Flags

- horizon_auxiliary_supported: `True`
- endpoint_linear_replacement_supported: `False`
- guarded_composer_supported: `True`
- ungated_full_waypoint_blocked: `True`
- unified_three_domain_row_cache_support: `True`
- source_support_closed: `False`

## Key Evidence

| evidence | all | t50 | t100 raw diagnostic | hard/failure | easy | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `endpoint_linear_bridge_floor` | 21.03% | 13.65% | 14.69% | 20.38% | -14.51% | current endpoint-linear / teacher protected floor |
| `protected_full_waypoint_transformer` | 18.58% | 14.80% | 22.86% | 19.52% | -0.00% | actual full-waypoint sequence model under protected switch |
| `ungated_full_waypoint_transformer` | 29.66% | 21.52% | 35.92% | 32.94% | 124.59% | unsafe diagnostic, blocked by easy degradation |
| `proximity_guarded_composer_vs_endpoint` | 1.77% | 1.07% | 3.48% | 1.93% | 0.25% | safety-sensitive guarded composer over endpoint-linear |
| `unified_row_cache_summary` | 9.00% | 6.11% | 8.15% | 9.37% | 0.11% | three-domain row-level full-waypoint cache support |

## Boundary Deltas

- full_waypoint_minus_endpoint_linear: `{'all_improvement': -0.02447280122019102, 't50_improvement': 0.011514681275772931, 't100_raw_frame_diagnostic_improvement': 0.08163339933561242, 'hard_failure_improvement': -0.008668690299824866}`
- graph_group_minus_full_waypoint: `{'all_improvement': 0.03662587747187018, 't50_improvement': 0.0029007439948979252, 't100_raw_frame_diagnostic_improvement': 0.0016194313330045729, 'hard_failure_improvement': 0.028937220874471148, 'collision_delta_vs_floor_005': 0.00829083972266037}`

## Proximity Guard Pareto Result

- no_proximity_guard: `{'name': 'no_proximity_guard', 'role': 'accuracy_priority_diagnostic', 'all_improvement': 0.030166976195252437, 't50_improvement': 0.01502943431774939, 't100_raw_frame_diagnostic_improvement': 0.06118732156780071, 'hard_failure_improvement': 0.03280214089079592, 'easy_degradation': 0.002532827637569346, 'switch_rate': 0.21349589396340585, 'near_collision_005_delta_vs_endpoint': 0.0033566895337319713, 'p05_min_distance_delta_vs_endpoint': -0.0015142309056838422, 'jagged_rate_delta_vs_endpoint': 0.0}`
- proximity_guard: `{'name': 'proximity_guard', 'role': 'safety_sensitive_deployable', 'all_improvement': 0.017743597342181783, 't50_improvement': 0.010673426149055754, 't100_raw_frame_diagnostic_improvement': 0.03480124336134305, 'hard_failure_improvement': 0.01929354722729537, 'easy_degradation': 0.0024927762207753723, 'switch_rate': 0.1696441434951736, 'near_collision_005_delta_vs_endpoint': -0.0006053046700172371, 'p05_min_distance_delta_vs_endpoint': -0.0001248494575282963, 'jagged_rate_delta_vs_endpoint': 0.0}`

## Next Training Targets

- train all-agent full-waypoint sequence model with all/hard ADE objective strong enough to beat endpoint-linear, not only t50/t100 raw-frame slices
- add proximity / collision / physical validity loss so graph or full-waypoint gains do not create proximity caveats
- keep validation-only domain/horizon safe-switch and easy-preservation loss; do not remove Stage37/teacher floor until no-floor gates pass
- replace current weak context/gain-harm protocol with richer joint occupancy or interaction-constraint target because Stage42-DC did not support context switchability
- close source/legal/time/t100 support for ETH_UCY, TrajNet, and UCY before broad metric/seconds/t100 claims

## Claim Boundary

- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'global_metric_claim_allowed': False, 'global_seconds_claim_allowed': False, 'full_waypoint_primary_deployable_claim_allowed': False, 'guarded_composer_claim_allowed': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Interpretation

- Stage42-DE closes the current full-waypoint deployment question honestly: full-waypoint is useful as a protected horizon/shape component, especially for t50/t100 raw-frame slices, but it is not yet promoted as the primary deployable world dynamics head.
- The safest deployable shape path remains a guarded composer under Stage37/teacher floor. Ungated full-waypoint remains unsafe.
- The next research move should change the training target/loss for all-agent full-waypoint dynamics, not relabel endpoint-linear success as full-waypoint success.
