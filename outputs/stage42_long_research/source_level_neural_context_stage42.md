# Stage42-AQ Proposed Source-Level Neural Context

- source: `fresh_run`
- generated_at_utc: `2026-05-26T08:44:52.272930+00:00`
- git_commit: `66b2b0e`
- input_hash: `be43d6a11049bb5b27a22959410827de6f42fcbded02ca62de979db0f80b3a04`
- gate: `11 / 12`
- verdict: `stage42_aq_neural_context_evidence_partial_or_negative`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AQ 是 proposed source-level split neural residual-context retraining，不是 metric 或 seconds-level 结果。
- 第一阶段只用 baseline-family rollout context；第二阶段用 PyTorch MLP 从 history/goal/neighbor context 预测 residual full-waypoint delta。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Runtime

- runtime: `{'python': '/Users/yangyue/Downloads/World/.venv-pytorch/bin/python', 'machine': 'arm64', 'torch_threads': 4, 'num_workers': 0, 'epochs': 3, 'batch': 8192, 'checkpoint_dir': 'outputs/stage42_long_research/checkpoints'}`

## Why This Was Run

- Stage42-AO/AP showed that ridge and ridge-residual context variants do not add >1% over baseline-family rollout context.
- Stage42-AQ replaces the second stage with a PyTorch MLP residual-context learner, keeping the same proposed source-level split and validation-only policy selection.

## Baseline-Family First Stage

- feature_count: `35`
- protected_metric: `{'rows': 47458, 'all_improvement': 0.2877734037648393, 't10_improvement': 0.47618706265543653, 't25_improvement': 0.31611808582214795, 't50_improvement': 0.31542535139554606, 't100_raw_frame_diagnostic_improvement': 0.14282475620015533, 'hard_failure_improvement': 0.2758122379367457, 'easy_degradation': -0.32418582524688455, 'switch_rate': 0.6606262379367019, 'harm_over_fallback': -0.13253112673847436}`

## Neural Residual Variants

| variant | alpha | all | t50 | t100 diag | hard/failure | easy | delta all | delta t50 | delta hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `neural_history` | 0.25 | 0.248711 | 0.227962 | 0.140677 | 0.233373 | -0.294274 | -0.039063 | -0.087464 | -0.042440 |
| `neural_goal_neighbor` | 0.25 | 0.246499 | 0.226151 | 0.144052 | 0.236859 | -0.268468 | -0.041275 | -0.089274 | -0.038954 |
| `neural_history_goal_neighbor` | 0.25 | 0.248860 | 0.223913 | 0.142297 | 0.234285 | -0.282959 | -0.038914 | -0.091513 | -0.041528 |

## Interpretation

- positive_neural_context_variants: `[]`
- neural_context_verdict: `stage42_aq_neural_context_not_supported`

- Stage42-AQ did not find tabular neural residual-context value beyond baseline-family rollout context.
- If negative, this does not prove history/goal/neighbor are useless; it means the next experiment should use graph/sequence/scene-rich context rather than a tabular MLP.
- Claims remain dataset-local raw-frame 2.5D, not metric/seconds-level, true-3D, foundation, Stage5C, or SMC evidence.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'family_fde_input': False, 'safe_strongest_idx_old_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
