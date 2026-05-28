# Stage42-IZ Source-Level Nonlinear Context Slice Audit

- source: `fresh_run_retrained_extra_trees_context_slice_audit`
- generated_at_utc: `2026-05-28T09:20:08.893464+00:00`
- git_commit: `b5f8110`
- input_hash: `f9c57b10ab7f8e1cd23b36cdfee1c94f0ab39e4bd8a3883ddb0b6ffc0d86f6f1`
- gate: `11 / 11`
- verdict: `stage42_iz_context_slice_audit_positive`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-IZ 是 Stage42-IY 后的切片级 context utility audit：检查非线性 context 是否只在特定 source / horizon / density / neighbor / curvature / goal-ambiguity 切片有效。
- 本阶段重新训练 sampled train-only ExtraTrees residual trials；validation 选 safe policy；test 只评一次。
- 所有切片阈值来自 train split quantiles，不用 test 调阈值。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals。
- 本阶段仍是 dataset-local/raw-frame 2.5D evidence，不是 metric 或 seconds-level 结果。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- decision: `context_has_powered_slice_level_support`
- supported_context_slice_count: `14`
- powered_slice_rows: `69`
- interpretation: At least one powered slice shows easy-safe lift over the nonlinear baseline-family reference; this is slice-limited context evidence, not a global context claim.

## Train-Only Slice Thresholds

| threshold | value |
| --- | ---: |
| `path_length_q75` | 5.896081 |
| `neighbor_count_q75` | 44.000000 |
| `min_neighbor_dist_q25` | 0.548786 |
| `density_q75` | 22.000000 |
| `ttc_q25` | 3.002832 |
| `curvature_q75` | 6.296901 |
| `abs_turn_angle_q75` | 4.109730 |
| `goal_ambiguity_q75` | 0.693129 |

## Best Powered Slice By Trial

| trial | slice | rows | delta all | delta t50 | delta t100 raw | delta hard | easy | supported |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tree_context_only_residual` | `domain_horizon:TrajNet|25` | 10770 | 0.183986 | 0.000000 | 0.000000 | 0.180963 | -0.414360 | `True` |
| `tree_full_residual` | `domain_horizon:TrajNet|25` | 10770 | 0.137802 | 0.000000 | 0.000000 | 0.141642 | -0.344928 | `True` |
| `tree_goal_neighbor_residual` | `easy` | 11192 | -0.142187 | -0.122657 | 0.128369 | -0.090580 | -0.000000 | `True` |

## Top Slice Rows

| slice | trial | rows | context all | baseline all | delta all | delta t50 | delta hard | easy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `domain_horizon:TrajNet|25` | `tree_context_only_residual` | 10770 | 0.283718 | 0.099732 | 0.183986 | 0.000000 | 0.180963 | -0.414360 |
| `horizon:25` | `tree_context_only_residual` | 13470 | 0.224608 | 0.078954 | 0.145654 | 0.000000 | 0.133570 | -0.346751 |
| `domain_horizon:TrajNet|25` | `tree_full_residual` | 10770 | 0.237534 | 0.099732 | 0.137802 | 0.000000 | 0.141642 | -0.344928 |
| `easy` | `tree_context_only_residual` | 11192 | 0.219673 | 0.142187 | 0.077486 | -0.049067 | 0.010592 | -0.219673 |
| `easy` | `tree_goal_neighbor_residual` | 11192 | 0.000000 | 0.142187 | -0.142187 | -0.122657 | -0.090580 | -0.000000 |
| `horizon:25` | `tree_full_residual` | 13470 | 0.188046 | 0.078954 | 0.109092 | 0.000000 | 0.104547 | -0.288648 |
| `easy` | `tree_full_residual` | 11192 | 0.200329 | 0.142187 | 0.058141 | -0.002306 | 0.016188 | -0.200329 |
| `domain_horizon:TrajNet|10` | `tree_full_residual` | 12342 | 0.500939 | 0.469619 | 0.031321 | 0.000000 | 0.038968 | -0.331854 |
| `horizon:10` | `tree_full_residual` | 15402 | 0.376591 | 0.353045 | 0.023546 | 0.000000 | 0.029182 | -0.285033 |
| `long_history_path` | `tree_context_only_residual` | 3584 | 0.239304 | 0.210765 | 0.028539 | -0.057634 | 0.025306 | -0.000000 |
| `long_history_path` | `tree_full_residual` | 3584 | 0.235822 | 0.210765 | 0.025058 | -0.059469 | 0.022915 | -0.000000 |
| `high_goal_ambiguity` | `tree_full_residual` | 16486 | 0.233502 | 0.219685 | 0.013817 | -0.054142 | 0.003413 | -0.223219 |
| `low_ttc` | `tree_full_residual` | 9380 | 0.230872 | 0.217377 | 0.013494 | -0.060951 | 0.004738 | -0.261151 |
| `domain_horizon:TrajNet|10` | `tree_context_only_residual` | 12342 | 0.473074 | 0.469619 | 0.003455 | 0.000000 | 0.012038 | -0.311810 |
| `high_density` | `tree_full_residual` | 103 | 0.311477 | 0.271315 | 0.040162 | -0.084230 | 0.028910 | -0.160313 |
| `high_neighbor_count` | `tree_full_residual` | 103 | 0.311477 | 0.271315 | 0.040162 | -0.084230 | 0.028910 | -0.160313 |
| `domain:TrajNet` | `tree_full_residual` | 37918 | 0.304827 | 0.295661 | 0.009166 | -0.081935 | -0.005396 | -0.236606 |
| `horizon:10` | `tree_context_only_residual` | 15402 | 0.355642 | 0.353045 | 0.002597 | 0.000000 | 0.009015 | -0.267818 |
| `all_test` | `tree_full_residual` | 47458 | 0.233701 | 0.226674 | 0.007027 | -0.064017 | -0.004100 | -0.200329 |
| `close_neighbor` | `tree_full_residual` | 8987 | 0.230838 | 0.226244 | 0.004595 | -0.061569 | -0.007693 | -0.201198 |

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_slice_thresholds': True, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Interpretation

- Stage42-IZ found slice-limited nonlinear context support. This can only be written as a narrow slice claim until broader validation passes.
- This closes another capacity/slice-level escape hatch for the current flattened context protocol; future work should change target/architecture/data context rather than repeat the same row-level residual setup.
