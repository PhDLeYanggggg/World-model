# Stage42-BQ Calibrated-Subset T50 Source-Family Support Repair

- source: `fresh_calibrated_subset_t50_support_repair`
- generated_at_utc: `2026-05-26T14:34:42.493792+00:00`
- git_commit: `23b6ea6`
- input_hash: `728190ab405838bf124af6104eb1cef45ea6873ffb1c0710a1b7bfbca4d7a212`
- gate: `12 / 12`
- verdict: `stage42_bq_calibrated_subset_t50_support_repair_pass_t50_nonharm_limited_positive`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BQ 只修复 Stage42-BP 暴露出的 calibrated-subset t50 source-family support blocker。
- t50 policy 只有同一 source-family 在 train+val 中至少两个独立支持源时才允许切换。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Repair Summary

- BP previous t50 min: `-0.06660887331981802`
- BP previous easy max: `-0.0`
- t50_min_source_family_support: `2`
- all_improvement_macro_mean: `0.04238031883856982`
- all_improvement_min: `0.0`
- t50_improvement_macro_mean: `0.0`
- t50_improvement_min: `0.0`
- hard_failure_improvement_macro_mean: `0.04026566086180111`
- easy_degradation_max: `-0.0`
- positive_fold_count: `3`
- positive_t50_fold_count: `0`

## Fold Results

| holdout | rows | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | policy slices |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_seq_eth` | 21598 | 0.108363 | 0.000000 | 0.023593 | 0.108376 | -0.298598 | 0.644087 | 3 |
| `ETH_seq_hotel` | 16611 | 0.042457 | 0.000000 | 0.078210 | 0.042790 | -0.060667 | 0.180423 | 2 |
| `UCY_students03` | 70585 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 4 |
| `UCY_zara01` | 16103 | 0.103462 | 0.000000 | 0.064975 | 0.090428 | -0.240122 | 0.594423 | 3 |
| `UCY_zara02` | 25901 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 3 |
| `UCY_zara03` | 9540 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 3 |

## Interpretation

- BQ tightens BP specifically for t50 by requiring at least two train+val sources in the same source-family before a t50 switch is allowed.
- This is a limited source-specific annotation-step calibrated subset repair, not a global metric/seconds-level M3W claim.
- Fallback-only sources are treated as unsupported rather than as positive transfer.

## Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'source_specific_annotation_step_subset_claim_allowed': True, 'global_metric_claim_allowed': False, 'global_seconds_claim_allowed': False, 'm3w_official_metric_seconds_claim_allowed': False, 'stage5c_executed': False, 'smc_enabled': False}`
