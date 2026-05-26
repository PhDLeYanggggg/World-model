# Stage42-AY AW T100 Easy-Safety Repair

- source: `fresh_run`
- generated_at_utc: `2026-05-26T11:09:17.292927+00:00`
- git_commit: `47cca91`
- input_hash: `4c8ce94f54f2c602eec54df9d1f53b652c4e20a9974f3e598e914675e0971a75`
- gate: `17 / 17`
- verdict: `stage42_ay_t100_easy_safety_repair_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AY 是 Stage42-AW repaired protocol 的 t100 easy-safety repair。
- 本修复重新计算 AW validation-best variant 的 full-waypoint arrays，并用 validation-only t100 easy-safety guard 决定是否回退 floor。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t100 仍是 raw-frame diagnostic，不是 seconds-level long-horizon claim。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Repair Policy

- validation_best_variant: `family_baseline_rel_only`
- best_lambda: `0.1`
- feature_count: `23`
- internal_val_group: `UCY::UCY/zara03/crowds_zara03.txt`
- threshold: `0.0`
- uses_test_metrics_for_threshold: `False`
- guarded_slices: `{'TrajNet|100': {'source': 'fresh_run_validation_only_t100_easy_guard', 'val_all_improvement': 0.23293241701548617, 'val_easy_degradation': 0.015205820485253652, 'threshold': 0.0, 'rows_all_splits': 17408, 'reason': 'validation_t100_easy_degradation_above_strict_nonharm_threshold_or_nonpositive_gain'}}`
- kept_slices: `{'ETH_UCY|100': {'source': 'fresh_run_validation_only_t100_easy_guard', 'val_all_improvement': 0.3653153268220918, 'val_easy_degradation': -0.05546129868680727, 'threshold': 0.0, 'rows_all_splits': 29328}, 'UCY|100': {'source': 'fresh_run_validation_only_t100_easy_guard', 'val_all_improvement': 0.2753989098404662, 'val_easy_degradation': -0.046127469252680964, 'threshold': 0.0, 'rows_all_splits': 10008}}`

## Before / After

| metric | before | after |
| --- | ---: | ---: |
| global all ADE improvement | 0.356806 | 0.305467 |
| h100 t100 raw-frame diagnostic | 0.210162 | 0.067836 |
| h100 easy degradation | 0.023961 | -0.006504 |

## Repaired Global Bootstrap

| metric | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.301698 | 0.305521 | 0.309282 | 47458 |
| `t50` | 0.284913 | 0.289754 | 0.294484 | 11538 |
| `t100_raw_frame_diagnostic` | 0.063247 | 0.067906 | 0.073089 | 7048 |
| `hard_failure` | 0.275586 | 0.279725 | 0.283976 | 35076 |
| `easy_degradation` | -0.616015 | -0.594413 | -0.572982 | 11192 |
| `h100_easy_degradation` | -0.022140 | -0.006941 | 0.009833 | 975 |

## Domain Metrics After Repair

| domain | rows | all | t50 | t100 raw diag | hard | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 0.284459 | 0.302119 | 0.000000 | 0.255974 | -0.364733 | 0.747824 |
| `UCY` | 9540 | 0.374492 | 0.245320 | 0.275399 | 0.355073 | -0.418376 | 0.632075 |

## Horizon Metrics After Repair

| horizon | rows | all | horizon metric | hard | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `10` | 15402 | 0.682892 | 0.682892 | 0.697541 | -0.505393 | 0.860992 |
| `25` | 13470 | 0.360850 | 0.360850 | 0.268682 | -0.502842 | 0.724722 |
| `50` | 11538 | 0.289698 | 0.289698 | 0.289698 | -0.214067 | 0.891229 |
| `100` | 7048 | 0.067836 | 0.067836 | 0.067836 | -0.006504 | 0.153235 |

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'internal_val_from_train_only': True, 'test_sources_unchanged': True, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 't100_seconds_claim': False, 'stage5c_executed': False, 'smc_enabled': False, 'ungated_neural_deployable': False, 'post_ax_repair_needs_future_holdout_for_stronger_paper_claim': True}`

## Interpretation

- Stage42-AY repairs the Stage42-AX horizon=100 easy-safety weakness with a validation-only strict t100 guard.
- This is a repaired-policy candidate after AX exposed the weak slice; stronger paper claims still need future held-out/source-level confirmation.
- t100 remains raw-frame diagnostic only; no metric, seconds-level, true-3D, Stage5C, SMC, or ungated-neural claim is made.
