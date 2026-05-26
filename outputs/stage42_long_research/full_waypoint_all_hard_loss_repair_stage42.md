# Stage42-DG Full-Waypoint All/Hard Weighted Loss Repair

- source: `fresh_stage42_dg_full_waypoint_all_hard_loss_repair`
- generated_at_utc: `2026-05-26T22:10:29.388069+00:00`
- git_commit: `f5927cb`
- gate: `13 / 15`
- verdict: `stage42_dg_full_waypoint_weighted_loss_repair_pass_positive_not_better_than_am`
- decision: `weighted_loss_not_enough_keep_stage42_am_or_cq_floor`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DG 针对 Stage42-DE/DF 的 full-waypoint all/hard blocker，实际重训 all/hard/long-horizon weighted full-waypoint ridge dynamics probe。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- validation 选择 loss variant、ridge lambda、safe policy；test 只评一次。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Selected Loss Variant

- variant: `balanced`
- lambda: `100.0`
- val_score: `2.376170`
- policy_slice_count: `8`

## Test Once vs Train-Horizon Causal Floor

| candidate | all | t50 | t100 raw diag | hard/failure | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ungated selected loss variant | 44.34% | 40.26% | 45.80% | 43.62% | -26.61% | 100.00% |
| protected selected loss variant | 24.58% | 22.02% | 14.37% | 23.75% | -25.66% | 58.81% |

## Delta vs Stage42-AM Protected Ridge

- delta_all: `0.00%`
- delta_t50: `0.00%`
- delta_t100_raw: `0.00%`
- delta_hard: `0.00%`
- delta_easy: `0.00%`

## Bootstrap CI

| slice | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.242756 | 0.245745 | 0.249142 | 47458 |
| `t50` | 0.215902 | 0.220090 | 0.223878 | 11538 |
| `t100_raw_frame_diagnostic` | 0.138350 | 0.143778 | 0.149566 | 7048 |
| `hard_failure` | 0.234138 | 0.237530 | 0.240794 | 35076 |
| `easy_degradation` | -0.359363 | -0.344727 | -0.331643 | 11192 |

## By Domain

| domain | rows | all | t50 | t100 raw diag | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 32.06% | 28.18% | 19.06% | 31.25% | -30.31% | 73.61% |
| `UCY` | 9540 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% |

## Interpretation

- Stage42-DG changes the full-waypoint training target through all/hard/long-horizon sample weighting and validation-selected ridge lambda.
- It is a real retraining/evaluation probe over source-level full-waypoint rows, not another Stage42-DF threshold search.
- Promotion requires improving Stage42-AM on all and hard/failure while keeping easy degradation <=2%.
- If it does not beat Stage42-AM, the next move should be a stronger sequence/graph model or explicit proximity/occupancy loss, not more scalar loss weights.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'validation_only_model_selection': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
