# Stage42-EZ Temporal Group-Repel Repair

- source: `fresh_stage42_temporal_group_repel_repair`
- generated_at_utc: `2026-05-27T05:34:42.763554+00:00`
- git_commit: `00e2bdb`
- gate: `17 / 18`
- verdict: `stage42_ez_temporal_group_repel_repair_positive_not_promoted`
- decision: `temporal_group_repel_not_enough_keep_stage42_di_or_cq_floor`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-EZ 接续 Stage42-EW/EX/EY 的负结果：risk/adaptive bucket 没有超过 Stage42-DI。
- 本阶段只改变 group-repel repair 的 temporal shape / candidate family，不启用 latent generative 或 SMC。
- repair 只使用 predicted rollout geometry、source/frame/horizon group key、agent id、当前/过去可得信息。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- candidate 和 policy 只在 validation 上选择；test 只评一次。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Repair Family

- candidate_count: `144`
- temporal_shapes: `['bell', 'head', 'sqrt_tail', 'tail', 'uniform']`
- direction_modes: `['centroid_current', 'nearest_current']`

## Selected Temporal Repair

- candidate: `{'mode': 'temporal_repel', 'temporal_kind': 'tail', 'gamma': 1.0, 'direction_mode': 'nearest_current', 'min_sep': 0.12, 'margin': 0.0, 'strength': 0.25}`
- val_score: `1.973020`
- val_metric: `{'rows': 23788, 'all_improvement': 0.44181323632456737, 't10_improvement': 0.6918594440695731, 't25_improvement': 0.30835298970387703, 't50_improvement': 0.47731083534869534, 't100_raw_frame_diagnostic_improvement': 0.32960035905504714, 'hard_failure_improvement': 0.4375194062732586, 'easy_degradation': -0.2877589884590561, 'switch_rate': 0.8334874726752984, 'harm_over_fallback': -0.23908667151341398}`
- val_diagnostics: `{'unsafe_rows': 3312, 'unsafe_rate': 0.13922986379687238, 'base_switch_rate': 0.8334874726752984, 'final_switch_rate': 0.8334874726752984, 'base_near_005': 0.015512022868673281, 'final_near_005': 0.005759206322515554, 'floor_near_005': 0.012737514713300825, 'base_p05_min_distance': 0.07308731743085398, 'final_p05_min_distance': 0.08333551348580309, 'floor_p05_min_distance': 0.0760265211798491, 'temporal_first_weight': 0.25, 'temporal_last_weight': 1.0}`

## Test Once Metrics vs Train-Horizon Causal Floor

| candidate | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | near@0.05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage42-AM rebuilt floor-protected source-level | 24.58% | 22.02% | 14.37% | 23.75% | -25.66% | 58.81% | 1.94% |
| Stage42-EZ temporal group-repel | 24.73% | 22.40% | 14.35% | 23.89% | -25.64% | 58.81% | 1.51% |

## Delta vs Stage42-DI

- all/t50/t100raw/hard/easy: `0.01%` / `0.04%` / `0.00%` / `0.00%` / `-0.01%`
- near_delta_vs_stage42_di: `0.13%`

## Bootstrap CI

| slice | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.244181 | 0.247165 | 0.250168 | 47458 |
| `t50` | 0.219304 | 0.223859 | 0.228056 | 11538 |
| `t100_raw_frame_diagnostic` | 0.138217 | 0.143404 | 0.149148 | 7048 |
| `hard_failure` | 0.235409 | 0.238855 | 0.242365 | 35076 |
| `easy_degradation` | -0.359536 | -0.345279 | -0.329828 | 11192 |

## By Domain

| domain | rows | all | t50 | t100 raw diag | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 32.25% | 28.67% | 19.04% | 31.44% | -30.29% | 73.61% |
| `UCY` | 9540 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% |

## Interpretation

- Stage42-EW/EX/EY showed that risk/adaptive bucket selection did not beat Stage42-DI.
- Stage42-EZ tests a different hypothesis: the constant same-offset repel is too crude, and temporal weighting might preserve early waypoints while repairing future group proximity.
- Promotion requires beating Stage42-DI on all and hard/failure while preserving easy and not worsening near@0.05.
- If not promoted, this is evidence that the bottleneck is not merely temporal offset shape; the next repair should change objective/trajectory family more deeply.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'group_features_predicted_rollout_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'validation_only_policy_selection': True, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
