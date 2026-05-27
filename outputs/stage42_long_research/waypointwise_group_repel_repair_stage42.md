# Stage42-FA Waypoint-Wise Group-Repel Repair

- source: `fresh_stage42_waypointwise_group_repel_repair`
- generated_at_utc: `2026-05-27T05:44:55.031541+00:00`
- git_commit: `47cd6a5`
- gate: `15 / 17`
- verdict: `stage42_fa_waypointwise_group_repel_repair_positive_not_promoted`
- decision: `waypointwise_group_repel_not_enough_keep_stage42_di_or_cq_floor`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-FA 接续 Stage42-EZ：simple temporal offset 有极小 ADE 增益但 proximity 比 Stage42-DI 差。
- 本阶段改为 waypoint-wise group repel：每个 future waypoint 按该时刻 predicted neighbor geometry 单独修复。
- repair 只使用 predicted rollout geometry、source/frame/horizon group key、agent id、当前/过去可得信息。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- candidate 和 policy 只在 validation 上选择；test 只评一次。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Repair Family

- candidate_count: `72`
- temporal_shapes: `['bell', 'sqrt_tail', 'tail', 'uniform']`
- smooth_options: `[False, True]`

## Selected Waypoint-Wise Repair

- candidate: `{'mode': 'waypointwise_repel', 'min_sep': 0.12, 'strength': 0.2, 'temporal_kind': 'sqrt_tail', 'gamma': 1.0, 'smooth': True, 'cap_scale': 0.75}`
- val_score: `1.969575`
- val_metric: `{'rows': 23788, 'all_improvement': 0.44145132875223214, 't10_improvement': 0.6933878964445332, 't25_improvement': 0.30500510238964884, 't50_improvement': 0.47467586548161633, 't100_raw_frame_diagnostic_improvement': 0.330481272704936, 'hard_failure_improvement': 0.4371705571886255, 'easy_degradation': -0.28724930871023147, 'switch_rate': 0.8334874726752984, 'harm_over_fallback': -0.23889082569044837}`
- val_diagnostics: `{'unsafe_rows': 3312, 'unsafe_rate': 0.13922986379687238, 'base_switch_rate': 0.8334874726752984, 'final_switch_rate': 0.8334874726752984, 'base_near_005': 0.015512022868673281, 'final_near_005': 0.004203800235412813, 'floor_near_005': 0.012737514713300825, 'base_p05_min_distance': 0.07308731743085398, 'final_p05_min_distance': 0.08494918554742388, 'floor_p05_min_distance': 0.0760265211798491, 'temporal_first_weight': 0.5, 'temporal_last_weight': 1.0}`

## Test Once Metrics vs Train-Horizon Causal Floor

| candidate | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | near@0.05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage42-AM rebuilt floor-protected source-level | 24.58% | 22.02% | 14.37% | 23.75% | -25.66% | 58.81% | 1.94% |
| Stage42-FA waypoint-wise group-repel | 24.61% | 22.05% | 14.36% | 23.77% | -25.67% | 58.81% | 1.21% |

## Delta vs Stage42-DI / Stage42-EZ

- delta_vs_stage42_di all/t50/t100raw/hard/easy: `-0.11%` / `-0.31%` / `0.02%` / `-0.11%` / `-0.03%`
- delta_vs_stage42_ez all/t50/t100raw/hard/easy: `-0.12%` / `-0.35%` / `0.02%` / `-0.12%` / `-0.02%`
- near_delta_vs_stage42_di: `-0.17%`
- near_delta_vs_stage42_ez: `-0.30%`

## Bootstrap CI

| slice | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.243009 | 0.246135 | 0.248994 | 47458 |
| `t50` | 0.216292 | 0.220496 | 0.224759 | 11538 |
| `t100_raw_frame_diagnostic` | 0.137772 | 0.143583 | 0.149197 | 7048 |
| `hard_failure` | 0.234293 | 0.237770 | 0.241144 | 35076 |
| `easy_degradation` | -0.359372 | -0.344582 | -0.332626 | 11192 |

## By Domain

| domain | rows | all | t50 | t100 raw diag | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 32.09% | 28.22% | 19.06% | 31.28% | -30.31% | 73.61% |
| `UCY` | 9540 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% |

## Interpretation

- Stage42-FA tests a stronger trajectory-family hypothesis than Stage42-EZ: each future waypoint receives a local group-consistency offset derived from same-time predicted neighbor geometry.
- Promotion requires beating Stage42-DI on all and hard/failure while preserving easy and not worsening near@0.05.
- If not promoted, this is evidence that group-consistency improvement needs a training/objective-level change rather than post-hoc repel family tweaks.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'group_features_predicted_rollout_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'validation_only_policy_selection': True, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
