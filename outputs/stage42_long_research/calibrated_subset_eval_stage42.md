# Stage42-BO Calibrated-Subset Source-CV Evaluation

- source: `fresh_calibrated_subset_source_cv`
- generated_at_utc: `2026-05-26T14:04:15.669924+00:00`
- git_commit: `31e3189`
- input_hash: `1d363dc34a97a3c640bb689dc821a4c735830975422d6ea63840cea4bdbf9b54`
- gate: `10 / 13`
- verdict: `stage42_bo_calibrated_subset_eval_partial`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BO 是 calibrated-subset-only source-CV evaluation，不是全局 metric 或 seconds-level claim。
- 只评估 Stage42-BN 审计出的 source-specific calibration candidates。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- calibrated_sources_evaluated: `6`
- source_cv_folds: `6`
- rows_total: `160338`
- all_improvement_macro_mean: `0.09050979339528516`
- all_improvement_min: `0.0`
- t50_improvement_macro_mean: `0.07072861716817978`
- t50_improvement_min: `-0.10778426225438564`
- t100_raw_frame_diagnostic_macro_mean: `0.10407147772840182`
- hard_failure_improvement_macro_mean: `0.09794392386813798`
- easy_degradation_max: `1.0325498116840084`
- positive_all_folds: `False`
- positive_t50_folds: `False`
- easy_safe_all_folds: `False`

## Fold Results

| holdout | val | rows | all | t50 | t100 raw diag | hard/failure | easy degradation | switch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_seq_eth` | `ETH_seq_hotel` | 21598 | 0.119101 | -0.107784 | 0.126728 | 0.119819 | -0.231576 | 0.842856 |
| `ETH_seq_hotel` | `UCY_students03` | 16611 | 0.030267 | 0.077718 | 0.000000 | 0.030350 | -0.073965 | 0.074770 |
| `UCY_students03` | `UCY_zara01` | 70585 | 0.032107 | 0.025788 | 0.157368 | 0.077401 | 1.032550 | 0.898109 |
| `UCY_zara01` | `UCY_zara02` | 16103 | 0.361583 | 0.428650 | 0.340333 | 0.360094 | -0.261642 | 0.778488 |
| `UCY_zara02` | `UCY_zara03` | 25901 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 |
| `UCY_zara03` | `ETH_seq_eth` | 9540 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 |

## Source Rows

| source | rows |
| --- | ---: |
| `ETH_seq_eth` | 21598 |
| `ETH_seq_hotel` | 16611 |
| `UCY_students03` | 70585 |
| `UCY_zara01` | 16103 |
| `UCY_zara02` | 25901 |
| `UCY_zara03` | 9540 |

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'source_specific_annotation_step_subset_claim_allowed': True, 'global_metric_claim_allowed': False, 'global_seconds_claim_allowed': False, 'm3w_official_metric_seconds_claim_allowed': False, 'raw_frame_dataset_local_global_claim_required': True, 'stage5c_executed': False, 'smc_enabled': False}`

## Interpretation

- This is a source-specific calibrated-subset evaluation candidate, not a global M3W metric/seconds-level claim.
- Fold splits are rebuilt at source level from Stage42-BN calibrated candidates; each source is held out once.
- Future waypoints/endpoints remain labels only, and threshold selection is validation-only.
- If a future paper uses this result, wording must restrict it to source-specific annotation-step calibrated subsets.
