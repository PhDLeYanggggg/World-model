# Stage42-JA Context-Slice Policy Promotion Audit

- source: `fresh_run_validation_selected_context_slice_policy`
- generated_at_utc: `2026-05-28T09:48:46.857696+00:00`
- git_commit: `7fe9871`
- input_hash: `d330253cc599ef0f1cc74715753dc05600d0d1d3e68f3ac05e9e1e58958ed881`
- gate: `10 / 12`
- verdict: `stage42_ja_context_slice_policy_not_promotable`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JA 接在 Stage42-IZ 后面：不再只问 context 切片是否存在，而是问 validation-selected context-slice policy 能否安全提升到 test。
- 切片阈值来自 train split quantiles；切片/模型选择只看 validation；test 只评一次。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调阈值。
- 本阶段仍是 dataset-local/raw-frame 2.5D evidence，不是 metric 或 seconds-level 结果。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- decision: `validation_selected_context_slice_policy_not_promoted`
- selected_rule_count: `13`
- test_context_rule_coverage_rate: `0.977327`
- context policy all/t50/t100raw/hard/easy: `0.203253` / `0.190761` / `0.107057` / `0.195825` / `-0.211871`
- baseline-family all/t50/t100raw/hard/easy: `0.226674` / `0.261494` / `0.191765` / `0.238710` / `-0.142187`
- delta vs baseline-family all/t50/t100raw/hard/easy: `-0.023421` / `-0.070733` / `-0.084708` / `-0.042885` / `-0.069684`

## Validation-Selected Context Rules

| slice | trial | val rows | val score | delta all | delta t50 | delta t100 raw | delta hard | easy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `domain_horizon:TrajNet|25` | `tree_context_only_residual` | 2175 | 0.361776 | 0.171717 | 0.000000 | 0.000000 | 0.171717 | -0.123479 |
| `horizon:25` | `tree_context_only_residual` | 6459 | 0.327595 | 0.148972 | 0.000000 | 0.000000 | 0.163842 | -0.303388 |
| `domain_horizon:ETH_UCY|25` | `tree_context_only_residual` | 4284 | 0.311944 | 0.139785 | 0.000000 | 0.000000 | 0.159297 | -0.441879 |
| `domain_horizon:TrajNet|50` | `tree_context_only_residual` | 1885 | 0.185317 | 0.049579 | 0.049579 | 0.000000 | 0.049579 | 0.006903 |
| `high_goal_ambiguity` | `tree_full_residual` | 2502 | 0.080457 | 0.013151 | 0.025193 | 0.027565 | 0.006756 | -0.393247 |
| `horizon:50` | `tree_context_only_residual` | 5873 | 0.071603 | 0.022169 | 0.022169 | 0.000000 | 0.022169 | -0.027196 |
| `domain_horizon:ETH_UCY|50` | `tree_full_residual` | 3988 | 0.065897 | 0.021224 | 0.021224 | 0.000000 | 0.021224 | -0.157318 |
| `high_curvature` | `tree_goal_neighbor_residual` | 1381 | 0.005450 | 0.001314 | 0.016860 | -0.031059 | -0.001137 | -0.000000 |
| `domain:ETH_UCY` | `tree_full_residual` | 16103 | 0.004597 | -0.008341 | 0.021224 | 0.006184 | -0.009581 | -0.305693 |
| `low_ttc` | `tree_full_residual` | 8069 | -0.016799 | -0.018771 | 0.016992 | 0.009847 | -0.018668 | -0.250877 |
| `long_history_path` | `tree_full_residual` | 10300 | -0.086442 | -0.035265 | 0.014241 | -0.016579 | -0.037405 | -0.266776 |
| `high_turn_angle` | `tree_full_residual` | 3683 | -0.178163 | -0.030124 | 0.046717 | -0.024903 | -0.037990 | 0.017692 |
| `domain:TrajNet` | `tree_context_only_residual` | 7685 | -0.714565 | -0.211060 | 0.049579 | -0.410687 | -0.211060 | -0.054770 |

## Test Breakdown

### By Domain

| domain | rows | all | t50 | t100 raw | hard | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 0.265111 | 0.244154 | 0.142045 | 0.257685 | -0.250240 | 0.695026 |
| `UCY` | 9540 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 |

### By Horizon

| horizon | rows | all | t50 | t100 raw | hard | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10` | 15402 | 0.364354 | 0.000000 | 0.000000 | 0.378280 | -0.264455 | 0.544670 |
| `25` | 13470 | 0.224608 | 0.000000 | 0.000000 | 0.208215 | -0.346751 | 0.740980 |
| `50` | 11538 | 0.190761 | 0.190761 | 0.000000 | 0.190761 | -0.073590 | 0.611111 |
| `100` | 7048 | 0.107057 | 0.000000 | 0.107057 | 0.107057 | 0.089058 | 0.132378 |

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'inference_safe_policy_slices': True, 'validation_only_rule_selection': True, 'train_only_slice_thresholds': True, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Interpretation

- Stage42-IZ local context evidence did not survive validation-selected promotion into a safe global test policy.
- If promotable, this remains a validation-selected protected slice policy; it is not a global context or ungated neural claim.
- If not promotable, Stage42-IZ stays as local evidence only and should not be used as a deployment claim.
