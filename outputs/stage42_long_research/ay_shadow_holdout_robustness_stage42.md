# Stage42-AZ AY Shadow-Holdout T100 Robustness Audit

- source: `fresh_run`
- generated_at_utc: `2026-05-26T11:20:50.791610+00:00`
- git_commit: `37c30f7`
- input_hash: `1466ceba073f05a442434e48cc1483202d1e1d200dfe2ba9805f20838b24d497`
- gate: `16 / 16`
- verdict: `stage42_az_shadow_holdout_robustness_pass_with_ay_t100_limitation`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AZ 是 Stage42-AY strict t100 guard 的 source-level shadow-holdout robustness audit。
- Shadow holdout 只从原 train sources 内部构建，不使用最终 test source 调参。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t100 仍是 raw-frame diagnostic，不是 seconds-level long-horizon claim。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Shadow Split

- split_plan: `{'source': 'fresh_run', 'rule': 'Use original train sources only. For each domain with >=3 t100-capable train sources, hold out the largest t100 source, use the second-largest as shadow validation, and fit on the rest.', 'final_val_test_excluded': True, 'domains': {'ETH_UCY': {'status': 'fresh_run', 'shadow_val_group': 'ETH_UCY::UCY/zara02/obsmat.txt', 'shadow_holdout_group': 'ETH_UCY::UCY/students03/obsmat.txt', 'shadow_train_groups': ['ETH_UCY::ETH/seq_eth/obsmat.txt', 'ETH_UCY::ETH/seq_hotel/obsmat.txt'], 't100_groups': [{'group': 'ETH_UCY::ETH/seq_hotel/obsmat.txt', 't100_rows': 2560}, {'group': 'ETH_UCY::ETH/seq_eth/obsmat.txt', 't100_rows': 2614}, {'group': 'ETH_UCY::UCY/zara02/obsmat.txt', 't100_rows': 5433}, {'group': 'ETH_UCY::UCY/students03/obsmat.txt', 't100_rows': 15470}]}, 'TrajNet': {'status': 'fresh_run', 'shadow_val_group': 'TrajNet::TrajNet/Train/crowds/crowds_zara02.txt', 'shadow_holdout_group': 'TrajNet::TrajNet/Train/crowds/students001.txt', 'shadow_train_groups': ['TrajNet::TrajNet/Test/biwi/biwi_eth.txt', 'TrajNet::TrajNet/Test/crowds/crowds_zara01.txt', 'TrajNet::TrajNet/Test/crowds/uni_examples.txt', 'TrajNet::TrajNet/Train/crowds/arxiepiskopi1.txt', 'TrajNet::TrajNet/Train/mot/PETS09-S2L1.txt'], 't100_groups': [{'group': 'TrajNet::TrajNet/Train/crowds/arxiepiskopi1.txt', 't100_rows': 480}, {'group': 'TrajNet::TrajNet/Train/crowds/crowds_zara02.txt', 't100_rows': 3032}, {'group': 'TrajNet::TrajNet/Train/crowds/students001.txt', 't100_rows': 7128}]}, 'UCY': {'status': 'not_run', 'reason': 'fewer_than_three_t100_capable_original_train_sources', 't100_groups': [{'group': 'UCY::UCY/zara03/crowds_zara03.txt', 't100_rows': 1440}, {'group': 'UCY::UCY/students01/students001-trajnet.txt', 't100_rows': 7128}]}}}`
- split_stats: `{'shadow_train': {'rows': 46186, 'domains': {'ETH_UCY': 38209, 'TrajNet': 7977}, 'sources': 7, 't100': 5654}, 'shadow_val': {'rows': 45988, 'domains': {'ETH_UCY': 25901, 'TrajNet': 20087}, 'sources': 2, 't100': 8465}, 'shadow_holdout': {'rows': 117808, 'domains': {'ETH_UCY': 70585, 'TrajNet': 47223}, 'sources': 2, 't100': 22598}, 'blocked_no_shadow_holdout': {'rows': 56763, 'domains': {'UCY': 56763}, 'sources': 2, 't100': 8568}, 'final_eval_excluded': {'rows': 71246, 'domains': {'ETH_UCY': 16103, 'TrajNet': 45603, 'UCY': 9540}, 'sources': 5, 't100': 11459}}`

## Model And Policies

- model: `{'source': 'fresh_run', 'variant_name': 'family_baseline_rel_only', 'feature_count': 23, 'best_lambda': 100.0, 'validation_score': 1.1040142439846499, 'validation_candidates': [{'lambda': 0.1, 'score': 1.082130601973314, 'policy_slices': ['ETH_UCY|10', 'ETH_UCY|100', 'ETH_UCY|25', 'ETH_UCY|50', 'TrajNet|10', 'TrajNet|25', 'TrajNet|50'], 'val_metric': {'rows': 45988, 'all_improvement': 0.27491375382077965, 't10_improvement': 0.654537809457084, 't25_improvement': 0.18086934453594172, 't50_improvement': 0.26499486393552507, 't100_raw_frame_diagnostic_improvement': 0.11231849663368165, 'hard_failure_improvement': 0.26599490036721585, 'easy_degradation': -0.20527778715241485, 'switch_rate': 0.5783682699834739, 'harm_over_fallback': -0.12839190475589}}, {'lambda': 1.0, 'score': 1.0822662736497215, 'policy_slices': ['ETH_UCY|10', 'ETH_UCY|100', 'ETH_UCY|25', 'ETH_UCY|50', 'TrajNet|10', 'TrajNet|25', 'TrajNet|50'], 'val_metric': {'rows': 45988, 'all_improvement': 0.2748857834291153, 't10_improvement': 0.6543467183047649, 't25_improvement': 0.18099186281032464, 't50_improvement': 0.2651196605879391, 't100_raw_frame_diagnostic_improvement': 0.11222711535461116, 'hard_failure_improvement': 0.26594453870545187, 'easy_degradation': -0.20594340268140843, 'switch_rate': 0.5783682699834739, 'harm_over_fallback': -0.128378841852297}}, {'lambda': 10.0, 'score': 1.0822416732251634, 'policy_slices': ['ETH_UCY|10', 'ETH_UCY|100', 'ETH_UCY|25', 'ETH_UCY|50', 'TrajNet|10', 'TrajNet|25', 'TrajNet|50'], 'val_metric': {'rows': 45988, 'all_improvement': 0.2745274676900341, 't10_improvement': 0.6527043246353876, 't25_improvement': 0.1820943742043749, 't50_improvement': 0.26569637365286114, 't100_raw_frame_diagnostic_improvement': 0.11144353439659238, 'hard_failure_improvement': 0.26536935229225145, 'easy_degradation': -0.2110353042625328, 'switch_rate': 0.5783682699834739, 'harm_over_fallback': -0.12821149904166912}}, {'lambda': 100.0, 'score': 1.1040142439846499, 'policy_slices': ['ETH_UCY|10', 'ETH_UCY|100', 'ETH_UCY|25', 'ETH_UCY|50', 'TrajNet|10', 'TrajNet|25', 'TrajNet|50'], 'val_metric': {'rows': 45988, 'all_improvement': 0.27608014013929383, 't10_improvement': 0.6498636521955692, 't25_improvement': 0.18674821359752636, 't50_improvement': 0.2766117767859392, 't100_raw_frame_diagnostic_improvement': 0.10827440101680497, 'hard_failure_improvement': 0.2661783027825022, 'easy_degradation': -0.23084823091482798, 'switch_rate': 0.5993085152648517, 'harm_over_fallback': -0.1289366376367806}}], 'policy_slices': ['ETH_UCY|10', 'ETH_UCY|100', 'ETH_UCY|25', 'ETH_UCY|50', 'TrajNet|10', 'TrajNet|25', 'TrajNet|50']}`
- ay_strict_t100_guard: `{'source': 'fresh_run', 'type': 'stage42ay_strict_validation_easy_guard_replayed_on_shadow_split', 'guarded_slices': {}, 'kept_slices': {'ETH_UCY|100': {'source': 'fresh_run_validation_only_t100_easy_guard', 'val_all_improvement': 0.16540904840428106, 'val_easy_degradation': -0.06572575789256918, 'threshold': 0.0, 'rows_all_splits': 29328}}, 'uses_final_test_metrics_for_threshold': False}`
- source_support_t100_guard: `{'source': 'fresh_run', 'type': 'shadow_source_support_t100_guard', 'min_t100_val_sources_per_domain': 2, 'guarded_slices': {'ETH_UCY|100': {'source': 'fresh_run_shadow_source_support_guard', 'val_t100_source_count': 1, 'min_t100_val_sources_per_domain': 2, 'val_all_improvement': 0.16540904840428106, 'val_easy_degradation': -0.06572575789256918, 'reason': 'insufficient_independent_t100_validation_sources_or_validation_easy_harm'}}, 'kept_slices': {}, 'uses_final_test_metrics_for_threshold': False}`

## AY Strict Guard On Shadow Holdout

| slice | rows | all | t50 | t100 raw diag | hard | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `global` | 117808 | 0.160719 | 0.121766 | 0.069499 | 0.158662 | -0.001709 | 0.659505 |
| `domain:ETH_UCY` | 70585 | 0.163965 | 0.133586 | 0.101482 | 0.166915 | 0.132161 | 0.789771 |
| `domain:TrajNet` | 47223 | 0.154597 | 0.100001 | 0.000000 | 0.141860 | -0.156388 | 0.464795 |
| `horizon:10` | 34363 | 0.470852 | 0.000000 | 0.000000 | 0.506808 | -0.140284 | 0.774001 |
| `horizon:25` | 31735 | 0.113546 | 0.000000 | 0.000000 | 0.102384 | -0.177534 | 0.631322 |
| `horizon:50` | 29112 | 0.121766 | 0.121766 | 0.000000 | 0.121766 | 0.173147 | 0.637950 |
| `horizon:100` | 22598 | 0.069499 | 0.000000 | 0.069499 | 0.069499 | 0.122946 | 0.552748 |

## Source-Support T100 Guard On Shadow Holdout

| slice | rows | all | t50 | t100 raw diag | hard | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `global` | 117808 | 0.133351 | 0.121766 | 0.000000 | 0.127756 | -0.022205 | 0.553477 |
| `domain:ETH_UCY` | 70585 | 0.122087 | 0.133586 | 0.000000 | 0.120829 | 0.093928 | 0.612807 |
| `domain:TrajNet` | 47223 | 0.154597 | 0.100001 | 0.000000 | 0.141860 | -0.156388 | 0.464795 |
| `horizon:10` | 34363 | 0.470852 | 0.000000 | 0.000000 | 0.506808 | -0.140284 | 0.774001 |
| `horizon:25` | 31735 | 0.113546 | 0.000000 | 0.000000 | 0.102384 | -0.177534 | 0.631322 |
| `horizon:50` | 29112 | 0.121766 | 0.121766 | 0.000000 | 0.121766 | 0.173147 | 0.637950 |
| `horizon:100` | 22598 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.000000 |

## Source-Support Guard Bootstrap

| metric | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.131153 | 0.133287 | 0.135447 | 117808 |
| `t50` | 0.116767 | 0.121706 | 0.126937 | 29112 |
| `t100_raw_frame_diagnostic` | 0.000000 | 0.000000 | 0.000000 | 22598 |
| `hard_failure` | 0.125429 | 0.127738 | 0.129972 | 86191 |
| `easy_degradation` | -0.029987 | -0.022321 | -0.015467 | 38255 |
| `h100_easy_degradation` | 0.000000 | 0.000000 | 0.000000 | 5922 |

## Summary

- summary: `{'source': 'fresh_run', 'ay_strict_guard_shadow_h100_easy_safe': False, 'source_support_guard_shadow_h100_easy_safe': True, 'source_support_guard_all_positive': True, 'source_support_guard_t50_positive': True, 'source_support_guard_t100_positive': False, 'source_support_guard_hard_positive': True, 'ucy_shadow_status': 'not_run', 'paper_claim': 'AY strict t100 guard is not independently robust on the original-train shadow holdout; ETH_UCY t100 easy harm appears. A more conservative source-support t100 guard protects easy cases but removes positive t100 gain on this shadow holdout.'}`

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'final_test_metrics_for_threshold': False, 'shadow_fit_val_holdout_from_original_train_only': True, 'train_only_feature_normalization': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 't100_seconds_claim': False, 'stage5c_executed': False, 'smc_enabled': False, 'ungated_neural_deployable': False}`

## Interpretation

- Stage42-AZ finds that the Stage42-AY strict t100 guard is not enough for original-train shadow-holdout robustness: ETH_UCY t100 easy harm appears.
- A conservative source-support guard protects shadow-holdout easy cases and keeps all/t50/hard positive, but it removes positive t100 gain on this shadow holdout.
- This is negative/repair evidence, not a new t100 success claim. It strengthens deployment safety boundaries and shows why t100 still needs more validation support.
