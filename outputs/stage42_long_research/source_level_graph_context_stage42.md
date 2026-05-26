# Stage42-AS Proposed Source-Level Graph Interaction Context

- source: `fresh_run`
- generated_at_utc: `2026-05-26T09:52:44.928325+00:00`
- git_commit: `8ccb3b6`
- input_hash: `8a22fc1607cf0b38dc6476a870078ef3cbd6fcdfb7c546c9e304303db42faf6f`
- gate: `10 / 11`
- verdict: `stage42_as_graph_context_evidence_partial_or_negative`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-AS 是 proposed source-level split graph-interaction residual training，不是 metric 或 seconds-level 结果。
- 第一阶段只用 baseline-family rollout context；第二阶段用 current-frame kNN graph / past motion / train-safe goal prototype context 预测 residual full-waypoint delta。
- graph features 只使用当前帧和过去 history，不使用 future endpoint / future waypoint 作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Why This Was Run

- Stage42-AQ ruled out a simple tabular MLP residual-context repair.
- Stage42-AR ruled out a temporal Conv1D sequence-context residual repair.
- Stage42-AS tests structured current-frame kNN graph / interaction context using only current and past information.

## Graph Schema

- graph_schema: `{'k_neighbors': 4, 'feature_names': ['graph_neighbor_count', 'graph_min_dist_norm', 'graph_mean_k_dist_norm', 'graph_inv_min_dist', 'graph_density_r1', 'graph_density_r2', 'graph_density_r5', 'graph_mean_speed_diff', 'graph_mean_heading_cos', 'graph_mean_heading_sin', 'graph_k0_dx_norm', 'graph_k0_dy_norm', 'graph_k0_dist_norm', 'graph_k0_closing_speed', 'graph_k0_heading_cos', 'graph_k0_heading_sin', 'graph_k1_dx_norm', 'graph_k1_dy_norm', 'graph_k1_dist_norm', 'graph_k1_closing_speed', 'graph_k1_heading_cos', 'graph_k1_heading_sin', 'graph_k2_dx_norm', 'graph_k2_dy_norm', 'graph_k2_dist_norm', 'graph_k2_closing_speed', 'graph_k2_heading_cos', 'graph_k2_heading_sin', 'graph_k3_dx_norm', 'graph_k3_dy_norm', 'graph_k3_dist_norm', 'graph_k3_closing_speed', 'graph_k3_heading_cos', 'graph_k3_heading_sin'], 'group_key': 'source_file + frame_id', 'excludes_self_agent': True, 'deduplicates_agent_horizon_rows': True, 'uses_current_and_past_only': True}`
- context_stats: `{'graph_only': {'source': 'fresh_run', 'rows': 337991, 'frame_groups': 11224, 'rows_with_neighbors': 334525, 'unique_agent_nodes_with_neighbors': 104125, 'max_unique_agents_per_frame': 65, 'feature_count': 34, 'uses_future_endpoint': False, 'uses_future_waypoint': False}, 'graph_goal': {'source': 'fresh_run', 'rows': 337991, 'frame_groups': 11224, 'rows_with_neighbors': 334525, 'unique_agent_nodes_with_neighbors': 104125, 'max_unique_agents_per_frame': 65, 'feature_count': 34, 'uses_future_endpoint': False, 'uses_future_waypoint': False}, 'graph_history_goal': {'source': 'fresh_run', 'rows': 337991, 'frame_groups': 11224, 'rows_with_neighbors': 334525, 'unique_agent_nodes_with_neighbors': 104125, 'max_unique_agents_per_frame': 65, 'feature_count': 34, 'uses_future_endpoint': False, 'uses_future_waypoint': False}}`

## Baseline-Family First Stage

- protected_metric: `{'rows': 47458, 'all_improvement': 0.2877734037648393, 't10_improvement': 0.47618706265543653, 't25_improvement': 0.31611808582214795, 't50_improvement': 0.31542535139554606, 't100_raw_frame_diagnostic_improvement': 0.14282475620015533, 'hard_failure_improvement': 0.2758122379367457, 'easy_degradation': -0.32418582524688455, 'switch_rate': 0.6606262379367019, 'harm_over_fallback': -0.13253112673847436}`

## Graph Residual Variants

| variant | lambda | alpha | features | all | t50 | t100 diag | hard/failure | easy | delta all | delta t50 | delta hard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `graph_only` | 100.00 | 0.25 | 41 | 0.264270 | 0.229673 | 0.142723 | 0.248810 | -0.317538 | -0.023503 | -0.085752 | -0.027002 |
| `graph_goal` | 0.10 | 0.25 | 51 | 0.264651 | 0.229035 | 0.142579 | 0.249472 | -0.316313 | -0.023123 | -0.086390 | -0.026341 |
| `graph_history_goal` | 0.10 | 0.25 | 60 | 0.264765 | 0.229008 | 0.142318 | 0.249578 | -0.316388 | -0.023009 | -0.086417 | -0.026235 |

## Interpretation

- positive_graph_context_variants: `[]`
- graph_context_verdict: `stage42_as_graph_context_not_supported`

- Stage42-AS did not find graph/interaction residual value beyond baseline-family rollout context.
- Claims remain dataset-local raw-frame 2.5D, not metric/seconds-level, true-3D, foundation, Stage5C, or SMC evidence.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'graph_features_current_and_past_only': True, 'family_fde_input': False, 'safe_strongest_idx_old_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'stage5c_executed': False, 'smc_enabled': False}`
