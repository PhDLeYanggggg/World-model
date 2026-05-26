# Stage42-BA Train-Only T100 Source-CV Repair

- source: `fresh_run`
- generated_at_utc: `2026-05-26T11:40:18.478205+00:00`
- git_commit: `f8efaf2`
- input_hash: `0f5fb58a8ebfa156461ffaae36726228364aa2673cfe4daebc96c837f974b0f3`
- gate: `16 / 16`
- verdict: `stage42_ba_t100_source_cv_repair_pass_with_t100_blocker`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BA 是 Stage42-AZ 后的 train-only t100 source-CV support audit / repair。
- source-CV folds 只从 original train sources 内部构建，final val/test 不参与 threshold 或 domain support 选择。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t100 仍是 raw-frame diagnostic，不是 seconds-level long-horizon claim。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Source-CV Plan

- plan: `{'source': 'fresh_run', 'rule': 'For each domain, leave one original-train t100-capable source out as source-CV holdout; choose the largest remaining t100-capable source as validation; train on all other original-train rows. Final val/test rows are excluded from support selection.', 'final_val_test_excluded': True, 'domains': {'ETH_UCY': {'status': 'fresh_run', 't100_groups': [{'group': 'ETH_UCY::UCY/students03/obsmat.txt', 't100_rows': 15470}, {'group': 'ETH_UCY::UCY/zara02/obsmat.txt', 't100_rows': 5433}, {'group': 'ETH_UCY::ETH/seq_eth/obsmat.txt', 't100_rows': 2614}, {'group': 'ETH_UCY::ETH/seq_hotel/obsmat.txt', 't100_rows': 2560}], 'fold_count': 4, 'folds': [{'source': 'fresh_run', 'domain': 'ETH_UCY', 'holdout_group': 'ETH_UCY::UCY/students03/obsmat.txt', 'holdout_t100_rows': 15470, 'val_group': 'ETH_UCY::UCY/zara02/obsmat.txt', 'train_groups': ['ETH_UCY::ETH/seq_eth/obsmat.txt', 'ETH_UCY::ETH/seq_hotel/obsmat.txt']}, {'source': 'fresh_run', 'domain': 'ETH_UCY', 'holdout_group': 'ETH_UCY::UCY/zara02/obsmat.txt', 'holdout_t100_rows': 5433, 'val_group': 'ETH_UCY::UCY/students03/obsmat.txt', 'train_groups': ['ETH_UCY::ETH/seq_eth/obsmat.txt', 'ETH_UCY::ETH/seq_hotel/obsmat.txt']}, {'source': 'fresh_run', 'domain': 'ETH_UCY', 'holdout_group': 'ETH_UCY::ETH/seq_eth/obsmat.txt', 'holdout_t100_rows': 2614, 'val_group': 'ETH_UCY::UCY/students03/obsmat.txt', 'train_groups': ['ETH_UCY::ETH/seq_hotel/obsmat.txt', 'ETH_UCY::UCY/zara02/obsmat.txt']}, {'source': 'fresh_run', 'domain': 'ETH_UCY', 'holdout_group': 'ETH_UCY::ETH/seq_hotel/obsmat.txt', 'holdout_t100_rows': 2560, 'val_group': 'ETH_UCY::UCY/students03/obsmat.txt', 'train_groups': ['ETH_UCY::ETH/seq_eth/obsmat.txt', 'ETH_UCY::UCY/zara02/obsmat.txt']}]}, 'TrajNet': {'status': 'fresh_run', 't100_groups': [{'group': 'TrajNet::TrajNet/Train/crowds/students001.txt', 't100_rows': 7128}, {'group': 'TrajNet::TrajNet/Train/crowds/crowds_zara02.txt', 't100_rows': 3032}, {'group': 'TrajNet::TrajNet/Train/crowds/arxiepiskopi1.txt', 't100_rows': 480}], 'fold_count': 3, 'folds': [{'source': 'fresh_run', 'domain': 'TrajNet', 'holdout_group': 'TrajNet::TrajNet/Train/crowds/students001.txt', 'holdout_t100_rows': 7128, 'val_group': 'TrajNet::TrajNet/Train/crowds/crowds_zara02.txt', 'train_groups': ['TrajNet::TrajNet/Test/biwi/biwi_eth.txt', 'TrajNet::TrajNet/Test/crowds/crowds_zara01.txt', 'TrajNet::TrajNet/Test/crowds/uni_examples.txt', 'TrajNet::TrajNet/Train/crowds/arxiepiskopi1.txt', 'TrajNet::TrajNet/Train/mot/PETS09-S2L1.txt']}, {'source': 'fresh_run', 'domain': 'TrajNet', 'holdout_group': 'TrajNet::TrajNet/Train/crowds/crowds_zara02.txt', 'holdout_t100_rows': 3032, 'val_group': 'TrajNet::TrajNet/Train/crowds/students001.txt', 'train_groups': ['TrajNet::TrajNet/Test/biwi/biwi_eth.txt', 'TrajNet::TrajNet/Test/crowds/crowds_zara01.txt', 'TrajNet::TrajNet/Test/crowds/uni_examples.txt', 'TrajNet::TrajNet/Train/crowds/arxiepiskopi1.txt', 'TrajNet::TrajNet/Train/mot/PETS09-S2L1.txt']}, {'source': 'fresh_run', 'domain': 'TrajNet', 'holdout_group': 'TrajNet::TrajNet/Train/crowds/arxiepiskopi1.txt', 'holdout_t100_rows': 480, 'val_group': 'TrajNet::TrajNet/Train/crowds/students001.txt', 'train_groups': ['TrajNet::TrajNet/Test/biwi/biwi_eth.txt', 'TrajNet::TrajNet/Test/crowds/crowds_zara01.txt', 'TrajNet::TrajNet/Test/crowds/uni_examples.txt', 'TrajNet::TrajNet/Train/crowds/crowds_zara02.txt', 'TrajNet::TrajNet/Train/mot/PETS09-S2L1.txt']}]}, 'UCY': {'status': 'not_run', 'reason': 'fewer_than_three_t100_capable_original_train_sources', 't100_groups': [{'group': 'UCY::UCY/students01/students001-trajnet.txt', 't100_rows': 7128}, {'group': 'UCY::UCY/zara03/crowds_zara03.txt', 't100_rows': 1440}]}}}`
- fold_count: `7`
- domain_support: `{'ETH_UCY': {'source': 'fresh_run', 'status': 'fresh_run', 'fold_count': 4, 'safe_positive_fold_count': 0, 'min_t100_improvement': 0.0, 'median_t100_improvement': 0.0, 'max_easy_degradation': 0.39783898629931813, 'supported_for_t100': False, 'support_rule': '>=2 safe-positive folds, min t100 > 0, max easy <= 0.02'}, 'TrajNet': {'source': 'fresh_run', 'status': 'fresh_run', 'fold_count': 3, 'safe_positive_fold_count': 1, 'min_t100_improvement': 0.15456595960382735, 'median_t100_improvement': 0.46707640500180736, 'max_easy_degradation': 0.0673639177746077, 'supported_for_t100': False, 'support_rule': '>=2 safe-positive folds, min t100 > 0, max easy <= 0.02'}, 'UCY': {'source': 'fresh_run', 'status': 'not_run', 'reason': 'fewer_than_three_t100_capable_original_train_sources', 'fold_count': 0, 'supported_for_t100': False}}`

## Source-CV Fold Results

| domain | holdout source | val source | t100 rows | best lambda | h100 t100 | h100 easy | safe positive | guarded | kept |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `ETH_UCY` | `ETH_UCY::UCY/students03/obsmat.txt` | `ETH_UCY::UCY/zara02/obsmat.txt` | 15470 | 100.000 | 0.135222 | 0.397839 | False | `[]` | `['ETH_UCY|100']` |
| `ETH_UCY` | `ETH_UCY::UCY/zara02/obsmat.txt` | `ETH_UCY::UCY/students03/obsmat.txt` | 5433 | 0.100 | 0.000000 | -0.000000 | False | `[]` | `[]` |
| `ETH_UCY` | `ETH_UCY::ETH/seq_eth/obsmat.txt` | `ETH_UCY::UCY/students03/obsmat.txt` | 2614 | 0.100 | 0.000000 | -0.000000 | False | `[]` | `[]` |
| `ETH_UCY` | `ETH_UCY::ETH/seq_hotel/obsmat.txt` | `ETH_UCY::UCY/students03/obsmat.txt` | 2560 | 0.100 | 0.000000 | -0.000000 | False | `[]` | `[]` |
| `TrajNet` | `TrajNet::TrajNet/Train/crowds/students001.txt` | `TrajNet::TrajNet/Train/crowds/crowds_zara02.txt` | 7128 | 10.000 | 0.154566 | -0.096950 | True | `[]` | `['TrajNet|100']` |
| `TrajNet` | `TrajNet::TrajNet/Train/crowds/crowds_zara02.txt` | `TrajNet::TrajNet/Train/crowds/students001.txt` | 3032 | 0.100 | 0.500353 | 0.067364 | False | `[]` | `['TrajNet|100']` |
| `TrajNet` | `TrajNet::TrajNet/Train/crowds/arxiepiskopi1.txt` | `TrajNet::TrajNet/Train/crowds/students001.txt` | 480 | 100.000 | 0.467076 | 0.054899 | False | `[]` | `['TrajNet|100']` |

## Final Test-On-Eval Guard

- guard: `{'source': 'fresh_run', 'type': 'stage42ba_train_source_cv_t100_guard', 'guarded_slices': {'ETH_UCY|100': {'source': 'fresh_run_train_source_cv_guard', 'reason': 'domain_lacks_train_source_cv_t100_support', 'support': {'source': 'fresh_run', 'status': 'fresh_run', 'fold_count': 4, 'safe_positive_fold_count': 0, 'min_t100_improvement': 0.0, 'median_t100_improvement': 0.0, 'max_easy_degradation': 0.39783898629931813, 'supported_for_t100': False, 'support_rule': '>=2 safe-positive folds, min t100 > 0, max easy <= 0.02'}, 'rows_all_splits': 29328}, 'TrajNet|100': {'source': 'fresh_run_train_source_cv_guard', 'reason': 'domain_lacks_train_source_cv_t100_support', 'support': {'source': 'fresh_run', 'status': 'fresh_run', 'fold_count': 3, 'safe_positive_fold_count': 1, 'min_t100_improvement': 0.15456595960382735, 'median_t100_improvement': 0.46707640500180736, 'max_easy_degradation': 0.0673639177746077, 'supported_for_t100': False, 'support_rule': '>=2 safe-positive folds, min t100 > 0, max easy <= 0.02'}, 'rows_all_splits': 17408}, 'UCY|100': {'source': 'fresh_run_train_source_cv_guard', 'reason': 'domain_lacks_train_source_cv_t100_support', 'support': {'source': 'fresh_run', 'status': 'not_run', 'reason': 'fewer_than_three_t100_capable_original_train_sources', 'fold_count': 0, 'supported_for_t100': False}, 'rows_all_splits': 10008}}, 'kept_slices': {}, 'uses_final_test_metrics_for_threshold': False}`
- supported_t100_domains: `[]`
- unsupported_t100_domains: `['ETH_UCY', 'TrajNet', 'UCY']`

## Before Source-CV Guard

| metric type | rows | all | t50 | t100 raw diag | hard | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ADE | 47458 | 0.356806 | 0.289698 | 0.210162 | 0.338904 | -0.370489 | 0.761242 |
| FDE | 47458 | 0.317131 | 0.269876 | 0.186636 | 0.297744 | -0.293176 | 0.761242 |

### Bootstrap

| slice | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.353215 | 0.356818 | 0.360442 | 47458 |
| `t50` | 0.284913 | 0.289754 | 0.294484 | 11538 |
| `t100_raw_frame_diagnostic` | 0.203628 | 0.210187 | 0.217099 | 7048 |
| `hard_failure` | 0.334970 | 0.338911 | 0.342627 | 35076 |
| `easy_degradation` | -0.611775 | -0.587853 | -0.566107 | 11192 |
| `h100_easy_degradation` | -0.003410 | 0.023520 | 0.053936 | 975 |

## After Source-CV Guard

| metric type | rows | all | t50 | t100 raw diag | hard | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ADE | 47458 | 0.280997 | 0.289698 | 0.000000 | 0.251576 | -0.372431 | 0.701799 |
| FDE | 47458 | 0.245628 | 0.269876 | 0.000000 | 0.215620 | -0.305268 | 0.701799 |

### Bootstrap

| slice | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.277271 | 0.280981 | 0.284608 | 47458 |
| `t50` | 0.284913 | 0.289754 | 0.294484 | 11538 |
| `t100_raw_frame_diagnostic` | 0.000000 | 0.000000 | 0.000000 | 7048 |
| `hard_failure` | 0.247811 | 0.251603 | 0.255290 | 35076 |
| `easy_degradation` | -0.614784 | -0.593163 | -0.571370 | 11192 |
| `h100_easy_degradation` | 0.000000 | 0.000000 | 0.000000 | 975 |

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'final_test_metrics_for_threshold': False, 'source_cv_from_original_train_only': True, 'train_only_feature_normalization': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 't100_seconds_claim': False, 'stage5c_executed': False, 'smc_enabled': False, 'ungated_neural_deployable': False}`

## Interpretation

- Stage42-BA uses train-only source-CV to decide whether any domain has enough independent t100 support.
- Unsupported t100 domain slices are guarded to the causal floor before final test evaluation.
- If t100 becomes zero after this guard, that is a safety/blocker result rather than a failure to report: it means current t100 positive gain lacks enough source-level validation support.
