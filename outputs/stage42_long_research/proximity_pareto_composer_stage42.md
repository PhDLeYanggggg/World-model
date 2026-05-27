# Stage42-FB Proximity Pareto Composer

- source: `fresh_stage42_proximity_pareto_composer`
- generated_at_utc: `2026-05-27T05:55:30.061542+00:00`
- git_commit: `0b8a830`
- gate: `14 / 16`
- verdict: `stage42_fb_proximity_pareto_composer_positive_not_promoted`
- decision: `proximity_pareto_composer_not_enough_keep_stage42_di_or_cq_floor`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DI 更准但 proximity 相对 FA 更差；Stage42-FA 更安全但 all/hard 低于 DI。
- Stage42-FB 用 validation-only composer 研究 DI/FA 的 safety-accuracy Pareto 边界。
- composer 只使用 predicted rollout geometry、source/frame/horizon group key、agent id、当前/过去可得信息。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- composer policy 只在 validation 上选择；test 只评一次。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Selected Composer

- candidate: `{'mode': 'group_di_near_fa_safer', 'threshold': 0.05, 'margin': 0.0}`
- val_score: `1.831564`
- val_metric: `{'rows': 23788, 'all_improvement': 0.44147556321447756, 't10_improvement': 0.6933195833166081, 't25_improvement': 0.3059255848095843, 't50_improvement': 0.4753909601795381, 't100_raw_frame_diagnostic_improvement': 0.32987337774741576, 'hard_failure_improvement': 0.43718188392690893, 'easy_degradation': -0.2872436150718558, 'switch_rate': 0.8334874726752984, 'harm_over_fallback': -0.2389039401389026}`
- val_diagnostics: `{'use_fa_rate': 0.016731124936942995, 'use_fa_rows': 398, 'di_near_005': 0.004371952244829325, 'fa_near_005': 0.004203800235412813, 'final_near_005': 0.0026904321506642003, 'di_p05_min_distance': 0.07964797237636109, 'fa_p05_min_distance': 0.08494918554742388, 'final_p05_min_distance': 0.07969842450780167}`

## Test Once Metrics vs Train-Horizon Causal Floor

| candidate | all | t50 | t100 raw diag | hard/failure | easy degradation | near@0.05 | use FA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage42-DI default | 24.72% | 22.36% | 14.35% | 23.89% | -25.63% | 1.38% | 0.00% |
| Stage42-FA safety | 24.61% | 22.05% | 14.36% | 23.77% | -25.67% | 1.21% | 100.00% |
| Stage42-FB composer | 24.65% | 22.19% | 14.35% | 23.82% | -25.64% | 1.10% | 9.34% |

## Delta

- delta_vs_DI all/t50/t100raw/hard/easy: `-0.07%` / `-0.18%` / `0.00%` / `-0.07%` / `-0.01%`
- delta_vs_FA all/t50/t100raw/hard/easy: `0.04%` / `0.13%` / `-0.01%` / `0.04%` / `0.03%`
- near_delta_vs_DI: `-0.29%`
- near_delta_vs_FA: `-0.11%`

## Bootstrap CI

| slice | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.243482 | 0.246452 | 0.249502 | 47458 |
| `t50` | 0.217525 | 0.221860 | 0.226066 | 11538 |
| `t100_raw_frame_diagnostic` | 0.137566 | 0.143564 | 0.149523 | 7048 |
| `hard_failure` | 0.234780 | 0.238177 | 0.241687 | 35076 |
| `easy_degradation` | -0.360387 | -0.345413 | -0.331807 | 11192 |

## By Domain

| domain | rows | all | t50 | t100 raw diag | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 32.15% | 28.40% | 19.04% | 31.34% | -30.28% | 73.61% |
| `UCY` | 9540 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% |

## Interpretation

- Stage42-FB explicitly tests whether the DI/FA Pareto boundary can be composed: use DI by default and FA only for predicted proximity-risk rows or groups.
- If promoted, it is a safety-sensitive runtime composer. If not, DI/CQ remains the floor and this documents the proximity/accuracy tradeoff.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'group_features_predicted_rollout_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'validation_only_policy_selection': True, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
