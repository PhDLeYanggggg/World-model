# Stage42-CK Neighbor/Interaction Gated Expert Audit

- source: `fresh_run`
- generated_at_utc: `2026-05-26T18:07:36.099291+00:00`
- git_commit: `c8abaf4`
- input_hash: `efedadc6dcce817deb27e7248fca62a48f3abd4d510a74d43a432e34d8a3a69f`
- gate: `11 / 11`
- verdict: `stage42_ck_neighbor_interaction_gated_expert_pass_diagnostic_no_overclaim`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CK 是 validation-only neighbor/interaction gated expert audit，不是 metric 或 seconds-level 结果。
- 本阶段专门测试 Stage42-CI 标出的 mixed/weak neighbor-interaction context 是否可被保守 gate 修复为增量贡献。
- 每个 candidate 都重新训练 ridge full-waypoint probe，并在 validation 上重新选 safe policy；test 只评一次。
- kNN graph features 只使用当前帧和过去 history，不使用 future endpoint / future waypoint 作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Why This Was Run

- Stage42-CI found neighbor/interaction context is mixed/weak and not a main claim.
- Stage42-AS had already shown graph residual context was not supported.
- Stage42-CK tests a stricter deployment question: can scalar neighbor or kNN graph features become a validation-only gated expert over baseline-family context?
- Test metrics are used only once after validation selection.

## Graph Schema

- graph_info: `{'graph_schema': {'k_neighbors': 4, 'feature_names': ['graph_neighbor_count', 'graph_min_dist_norm', 'graph_mean_k_dist_norm', 'graph_inv_min_dist', 'graph_density_r1', 'graph_density_r2', 'graph_density_r5', 'graph_mean_speed_diff', 'graph_mean_heading_cos', 'graph_mean_heading_sin', 'graph_k0_dx_norm', 'graph_k0_dy_norm', 'graph_k0_dist_norm', 'graph_k0_closing_speed', 'graph_k0_heading_cos', 'graph_k0_heading_sin', 'graph_k1_dx_norm', 'graph_k1_dy_norm', 'graph_k1_dist_norm', 'graph_k1_closing_speed', 'graph_k1_heading_cos', 'graph_k1_heading_sin', 'graph_k2_dx_norm', 'graph_k2_dy_norm', 'graph_k2_dist_norm', 'graph_k2_closing_speed', 'graph_k2_heading_cos', 'graph_k2_heading_sin', 'graph_k3_dx_norm', 'graph_k3_dy_norm', 'graph_k3_dist_norm', 'graph_k3_closing_speed', 'graph_k3_heading_cos', 'graph_k3_heading_sin']}, 'graph_stats': {'source': 'fresh_run', 'rows': 337991, 'frame_groups': 11224, 'rows_with_neighbors': 334525, 'unique_agent_nodes_with_neighbors': 104125, 'max_unique_agents_per_frame': 65, 'feature_count': 34, 'uses_future_endpoint': False, 'uses_future_waypoint': False}}`

## Variant Metrics

| variant | features | val score | all | t50 | t100 diag | hard/failure | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_family_control` | 35 | 2.007275 | 0.287773 | 0.315425 | 0.142825 | 0.275812 | -0.324186 | 0.660626 |
| `baseline_plus_scalar_neighbor` | 40 | 1.924754 | 0.263747 | 0.229616 | 0.143182 | 0.248791 | -0.313188 | 0.628977 |
| `baseline_plus_goal_scene` | 45 | 1.901516 | 0.262502 | 0.227640 | 0.142748 | 0.248648 | -0.301712 | 0.633739 |
| `baseline_plus_history_scalar` | 44 | 1.917642 | 0.263336 | 0.229490 | 0.142962 | 0.248381 | -0.312964 | 0.629083 |
| `baseline_plus_knn_graph` | 69 | 1.899381 | 0.243823 | 0.223773 | 0.143873 | 0.237773 | -0.256423 | 0.595179 |
| `baseline_plus_graph_goal` | 79 | 1.851584 | 0.206748 | 0.222053 | 0.144621 | 0.188102 | -0.272265 | 0.447343 |
| `baseline_plus_graph_history_scalar` | 78 | 1.869512 | 0.195701 | 0.223969 | 0.143665 | 0.184603 | -0.244151 | 0.462177 |

## Validation-Only Selection

- selection: `{'source': 'fresh_run', 'selection_rule': 'choose_neighbor_interaction_candidate_only_if_validation_score_beats_baseline_by_margin_and_easy_safe_else_fallback', 'validation_margin': 0.01, 'easy_limit': 0.02, 'baseline_variant': 'baseline_family_control', 'selected_variant': 'baseline_family_control', 'selected_score': 2.007274993106131, 'considered_neighbor_interaction_candidates': [{'variant': 'baseline_plus_scalar_neighbor', 'validation_score': 1.924753737427354, 'validation_margin_vs_baseline': -0.08252125567877688, 'validation_easy_degradation': -0.3262202630351835, 'passes_validation_gate': False}, {'variant': 'baseline_plus_knn_graph', 'validation_score': 1.8993807086353283, 'validation_margin_vs_baseline': -0.10789428447080263, 'validation_easy_degradation': -0.2593931572307454, 'passes_validation_gate': False}, {'variant': 'baseline_plus_graph_goal', 'validation_score': 1.8515844860908335, 'validation_margin_vs_baseline': -0.1556905070152974, 'validation_easy_degradation': -0.2648162813752213, 'passes_validation_gate': False}, {'variant': 'baseline_plus_graph_history_scalar', 'validation_score': 1.8695118823461143, 'validation_margin_vs_baseline': -0.13776311076001657, 'validation_easy_degradation': -0.26674402883087356, 'passes_validation_gate': False}], 'test_threshold_tuning': False}`
- selected_delta_vs_baseline_family_control: `{'all_improvement': 0.0, 't50_improvement': 0.0, 't100_raw_frame_diagnostic_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'switch_rate': 0.0, 'harm_over_fallback': 0.0}`
- neighbor_interaction_rescue_success: `False`

## Delta Vs Baseline-Family Control

| variant | delta all | delta t50 | delta hard/failure | delta easy |
| --- | ---: | ---: | ---: | ---: |
| `baseline_plus_scalar_neighbor` | -0.024026 | -0.085810 | -0.027021 | 0.010997 |
| `baseline_plus_goal_scene` | -0.025271 | -0.087785 | -0.027164 | 0.022474 |
| `baseline_plus_history_scalar` | -0.024437 | -0.085935 | -0.027431 | 0.011222 |
| `baseline_plus_knn_graph` | -0.043950 | -0.091653 | -0.038039 | 0.067763 |
| `baseline_plus_graph_goal` | -0.081026 | -0.093372 | -0.087710 | 0.051921 |
| `baseline_plus_graph_history_scalar` | -0.092072 | -0.091456 | -0.091209 | 0.080034 |

## Interpretation

- verdict: `neighbor_interaction_gated_expert_not_validation_selected`
- summary: The validation-only gate did not select a neighbor/interaction candidate over the baseline-family control. This preserves the Stage42-CI boundary: neighbor/interaction remains mixed/weak and not a main claim.

## No-Leakage And Claim Boundary

- no_leakage: `{'future_endpoint_input': False, 'future_waypoint_input': False, 'future_waypoint_label_eval_only': True, 'graph_features_current_and_past_only': True, 'family_fde_input': False, 'safe_strongest_idx_old_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'train_only_feature_normalization': True, 'source_overlap_pass': True}`
- claim_boundary: `{'true_3d': False, 'foundation_world_model': False, 'metric_or_seconds_claim': False, 'raw_frame_dataset_local_only': True, 'neighbor_interaction_main_claim_allowed': False, 'stage5c_executed': False, 'smc_enabled': False}`
