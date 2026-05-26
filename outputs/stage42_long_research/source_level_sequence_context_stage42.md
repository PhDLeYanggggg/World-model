# Stage42-AR Proposed Source-Level Sequence Context

- source: `fresh_run`
- generated_at_utc: `2026-05-26T09:14:48.649520+00:00`
- git_commit: `6720e8f`
- input_hash: `44f3603b226d542241da8571329b7b1c8c628941f94328a110c32aa74bd9ca0b`
- gate: `11 / 12`
- verdict: `stage42_ar_sequence_context_evidence_partial_or_negative`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AR 是 proposed source-level split sequence-context residual training，不是 metric 或 seconds-level 结果。
- 第一阶段只用 baseline-family rollout context；第二阶段用 temporal sequence encoder 和 goal/neighbor context 预测 residual full-waypoint delta。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Runtime And Schema

- runtime: `{'python': '/Users/yangyue/Downloads/World/.venv-pytorch/bin/python', 'machine': 'arm64', 'torch_threads': 4, 'num_workers': 0, 'epochs': 2, 'batch': 4096, 'checkpoint_dir': 'outputs/stage42_long_research/checkpoints'}`
- sequence_schema: `{'history_seq_shape': [337991, 64, 7], 'uses_past_history_only': True, 'goal_neighbor_context': ['prototype_likelihood', 'prototype_entropy', 'goal_ambiguity', 'history_scalar_neighbor_slice']}`

## Why This Was Run

- Stage42-AQ ruled out a simple tabular MLP residual-context repair.
- Stage42-AR uses a temporal Conv1D sequence encoder over past-only history plus goal/neighbor context.

## Baseline-Family First Stage

- protected_metric: `{'rows': 47458, 'all_improvement': 0.2877734037648393, 't10_improvement': 0.47618706265543653, 't25_improvement': 0.31611808582214795, 't50_improvement': 0.31542535139554606, 't100_raw_frame_diagnostic_improvement': 0.14282475620015533, 'hard_failure_improvement': 0.2758122379367457, 'easy_degradation': -0.32418582524688455, 'switch_rate': 0.6606262379367019, 'harm_over_fallback': -0.13253112673847436}`

## Sequence Residual Variants

| variant | alpha | all | t50 | t100 diag | hard/failure | easy | delta all | delta t50 | delta hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sequence_history` | 0.25 | 0.263315 | 0.232368 | 0.141711 | 0.247414 | -0.309828 | -0.024458 | -0.083057 | -0.028398 |
| `sequence_goal_neighbor_no_history` | 0.25 | 0.261300 | 0.223289 | 0.141222 | 0.246740 | -0.286020 | -0.026473 | -0.092136 | -0.029072 |
| `sequence_history_goal_neighbor` | 0.25 | 0.200250 | 0.227897 | 0.139987 | 0.179724 | -0.270441 | -0.087523 | -0.087529 | -0.096088 |

## Interpretation

- positive_sequence_context_variants: `[]`
- sequence_context_verdict: `stage42_ar_sequence_context_not_supported`

- Stage42-AR did not find sequence-context residual value beyond baseline-family rollout context.
- Claims remain dataset-local raw-frame 2.5D, not metric/seconds-level, true-3D, foundation, Stage5C, or SMC evidence.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'family_fde_input': False, 'safe_strongest_idx_old_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
