# Stage42-DH Full-Waypoint Proximity / Occupancy-Proxy Loss Repair

- source: `fresh_stage42_dh_full_waypoint_proximity_occupancy_loss_repair`
- generated_at_utc: `2026-05-26T22:19:47.990354+00:00`
- git_commit: `424881c`
- gate: `15 / 16`
- verdict: `stage42_dh_proximity_occupancy_loss_repair_pass_positive_not_better_than_am`
- decision: `proximity_occupancy_loss_not_enough_keep_stage42_am_or_cq_floor`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DH 针对 Stage42-DE 的 proximity / interaction blocker，实际重训 proximity/occupancy-proxy weighted full-waypoint ridge dynamics probe。
- graph/proximity/occupancy-proxy signals 只来自当前帧和过去 history，不使用 future endpoint / future waypoint 作为 inference input。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- validation 选择 loss variant、ridge lambda、safe policy；test 只评一次。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Selected Candidate

- variant: `proximity_close_weighted`
- feature_mode: `stage42_am_features`
- lambda: `100.0`
- val_score: `2.416932`
- policy_slice_count: `8`
- mean_train_weight: `1.000000`
- max_train_weight: `1.215170`

## Graph / Proximity Schema

- graph_stats: `{'source': 'fresh_run', 'rows': 337991, 'frame_groups': 11224, 'rows_with_neighbors': 334525, 'unique_agent_nodes_with_neighbors': 104125, 'max_unique_agents_per_frame': 65, 'feature_count': 34, 'uses_future_endpoint': False, 'uses_future_waypoint': False}`

## Test Once vs Train-Horizon Causal Floor

| candidate | all | t50 | t100 raw diag | hard/failure | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ungated selected candidate | 44.39% | 39.96% | 45.60% | 43.62% | -26.64% | 100.00% |
| protected selected candidate | 25.51% | 22.14% | 14.34% | 23.74% | -29.23% | 58.46% |

## Delta vs Stage42-AM Protected Ridge

- delta_all: `0.93%`
- delta_t50: `0.12%`
- delta_t100_raw: `-0.03%`
- delta_hard: `-0.01%`
- delta_easy: `-3.57%`

## Bootstrap CI

| slice | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.251868 | 0.255015 | 0.258319 | 47458 |
| `t50` | 0.217161 | 0.221398 | 0.225634 | 11538 |
| `t100_raw_frame_diagnostic` | 0.138026 | 0.143500 | 0.149080 | 7048 |
| `hard_failure` | 0.233895 | 0.237426 | 0.240671 | 35076 |
| `easy_degradation` | -0.430863 | -0.413058 | -0.396846 | 11192 |

## By Domain

| domain | rows | all | t50 | t100 raw diag | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet` | 37918 | 33.27% | 28.33% | 19.03% | 31.24% | -34.52% | 73.17% |
| `UCY` | 9540 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% |

## Interpretation

- Stage42-DH changes the training target toward current/past proximity and density/occupancy-proxy emphasis.
- It is a fresh retraining/evaluation probe; future waypoints remain labels only.
- Promotion requires improving Stage42-AM on all and hard/failure while keeping easy degradation <=2%.
- If it does not beat Stage42-AM, proximity/occupancy proxy weighting alone is not enough; the next move should use an explicit group-consistency/proximity differentiable model or stronger all-agent sequence architecture.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'graph_features_current_and_past_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'validation_only_model_selection': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
