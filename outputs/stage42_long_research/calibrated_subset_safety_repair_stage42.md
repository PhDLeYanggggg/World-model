# Stage42-BP Calibrated-Subset Safety Repair

- source: `fresh_calibrated_subset_safety_repair`
- generated_at_utc: `2026-05-26T14:16:08.517667+00:00`
- git_commit: `31e3189`
- input_hash: `c9ea6fc8bb778547d038ac36d74ce252bfdbb653452e989a58ce0d28360ce235`
- gate: `11 / 11`
- verdict: `stage42_bp_calibrated_subset_safety_repair_pass_limited_positive`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BP 修复 Stage42-BO calibrated-subset source-CV 的 easy/t50 failure，不使用 test 调阈值。
- 本步骤只允许 train+val support source/source-family guard，holdout source 只最终评估一次。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## What Failed In BO

- BO all_improvement_min: `0.0`
- BO t50_improvement_min: `-0.10778426225438564`
- BO easy_degradation_max: `1.0325498116840084`

## Repair Summary

- source_cv_folds: `6`
- all_improvement_macro_mean: `0.057580173663012824`
- all_improvement_min: `0.0`
- t50_improvement_macro_mean: `0.06186813927682248`
- t50_improvement_min: `-0.06660887331981802`
- hard_failure_improvement_macro_mean: `0.05628165647182785`
- easy_degradation_max: `-0.0`
- positive_fold_count: `3`
- positive_t50_fold_count: `2`

## Fold Results

| holdout | rows | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | policy slices |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_seq_eth` | 21598 | 0.094803 | -0.066609 | 0.025879 | 0.093957 | -0.259789 | 0.888369 | 4 |
| `ETH_seq_hotel` | 16611 | 0.068678 | 0.112582 | 0.078210 | 0.069408 | -0.061842 | 0.275540 | 3 |
| `UCY_students03` | 70585 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 4 |
| `UCY_zara01` | 16103 | 0.182000 | 0.325235 | 0.064975 | 0.174325 | -0.299710 | 0.808359 | 4 |
| `UCY_zara02` | 25901 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 4 |
| `UCY_zara03` | 9540 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | 3 |

## Interpretation

- BP adds train+val source-robust easy/harm guards to repair BO's calibrated-subset over-switching.
- This remains a limited source-specific annotation-step calibrated subset result, not global metric/seconds-level M3W.
- If a source still falls back to zero, it is treated as unsupported rather than overclaimed.

## Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'source_specific_annotation_step_subset_claim_allowed': True, 'global_metric_claim_allowed': False, 'global_seconds_claim_allowed': False, 'm3w_official_metric_seconds_claim_allowed': False, 'raw_frame_dataset_local_global_claim_required': True, 'stage5c_executed': False, 'smc_enabled': False}`
