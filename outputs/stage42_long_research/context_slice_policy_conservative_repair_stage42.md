# Stage42-JB Conservative Context-Slice Policy Repair

- source: `fresh_run_validation_greedy_conservative_context_slice_repair`
- generated_at_utc: `2026-05-28T11:33:48.216543+00:00`
- git_commit: `d93c90b`
- input_hash: `21a638fa3a1e0fc2d2274065e23be68b466f8a569c45267d4ba0bd5eea7adc52`
- gate: `11 / 13`
- verdict: `stage42_jb_conservative_context_policy_not_promotable`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-JB 是 Stage42-JA 失败后的修复实验：只允许 validation-greedy、inference-safe、core-preserving 的 context slice 切换。
- 切片阈值来自 train split quantiles；候选排序和贪心选择只看 validation；test 只评一次。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调阈值。
- 本阶段仍是 dataset-local/raw-frame 2.5D evidence，不是 metric 或 seconds-level 结果。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- decision: `conservative_context_slice_policy_not_promoted`
- selected_rule_count: `4`
- test_context_rule_coverage_rate: `0.526950`
- context policy all/t50/t100raw/hard/easy: `0.231382` / `0.190761` / `0.191765` / `0.227164` / `-0.220374`
- baseline-family all/t50/t100raw/hard/easy: `0.226674` / `0.261494` / `0.191765` / `0.238710` / `-0.142187`
- delta vs baseline-family all/t50/t100raw/hard/easy: `0.004708` / `-0.070733` / `0.000000` / `-0.011546` / `-0.078187`
- primary_blocker: `context_policy_has_core_metric_regression`

## Validation-Greedy Accepted Rules

| slice | trial | val rows | score before | score after | d all | d t50 | d t100 raw | d hard | d easy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `domain_horizon:TrajNet|25` | `tree_context_only_residual` | 2175 | 2.184074 | 2.198773 | 0.006419 | 0.000000 | 0.000000 | 0.006733 | 0.026012 |
| `horizon:25` | `tree_context_only_residual` | 4284 | 2.198773 | 2.226201 | 0.012937 | 0.000000 | 0.000000 | 0.010822 | 0.012106 |
| `domain_horizon:TrajNet|50` | `tree_context_only_residual` | 1885 | 2.226201 | 2.256420 | 0.003045 | 0.012806 | 0.000000 | 0.003194 | 0.008473 |
| `horizon:50` | `tree_context_only_residual` | 3988 | 2.256420 | 2.278514 | 0.002226 | 0.009363 | 0.000000 | 0.002335 | 0.033102 |

## Test Breakdown

### By Domain

| domain | rows | all | t50 | t100 raw | hard | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 0.301801 | 0.244154 | 0.254438 | 0.298924 | -0.260282 | 0.779261 |
| `UCY` | 9540 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 |

### By Horizon

| horizon | rows | all | t50 | t100 raw | hard | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10` | 15402 | 0.353045 | 0.000000 | 0.000000 | 0.359503 | -0.331586 | 0.705363 |
| `25` | 13470 | 0.224608 | 0.000000 | 0.000000 | 0.208215 | -0.346751 | 0.740980 |
| `50` | 11538 | 0.190761 | 0.190761 | 0.000000 | 0.190761 | -0.073590 | 0.611111 |
| `100` | 7048 | 0.191765 | 0.000000 | 0.191765 | 0.191765 | 0.128369 | 0.234393 |

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'validation_only_greedy_selection': True, 'inference_safe_policy_slices': True, 'train_only_slice_thresholds': True, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Interpretation

- Conservative validation-greedy repair still cannot promote context slices over the baseline-family mechanism.
- A promotable result would support context slices only as a guarded, validation-selected policy. A negative result means Stage42-IZ remains local analysis evidence only.
