# Stage42-IO Horizon-Specific Sequence+Graph Context Router

- source: `fresh_stage42_horizon_sequence_graph_context_router`
- generated_at_utc: `2026-05-29T04:01:45.901801+00:00`
- git_commit: `0ca07d9`
- input_hash: `7a2bf47de6031d5b1421d8d86fcc6278cc85dbb5e451550ad0156064cd8208be`
- gate: `13 / 13`
- verdict: `stage42_io_horizon_sequence_graph_context_router_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-IO 是 horizon-specific sequence+graph context router，不是新数据转换或 metric/seconds-level 结果。
- 该实验修复 Stage42-EQ 的 horizon mixing 风险：t10/t25/t50/t100 分开训练 gain/harm router，再在 test 上评一次。
- sequence summary 与 graph summary 只使用当前帧和过去 history。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- router_target: `horizon-specific supervised gain/harm routing for context proposal over baseline-family protected control`
- horizons: `[10, 25, 50, 100]`
- candidates: `['history_only', 'motion_goal_context', 'baseline_plus_history_goal_neighbor']`
- positive_horizon_sequence_graph_context_routers: `['h10_history_only', 'h10_motion_goal_context', 'h25_baseline_plus_history_goal_neighbor']`
- best_overall_router: `h10_motion_goal_context`
- horizon_specific_increment_verdict: `stage42_io_horizon_sequence_graph_context_router_supported`

Stage42-IO separates horizons after Stage42-EQ's global sequence+graph router failed. A positive row supports only a narrow horizon-specific routing contribution. If no horizon is positive, horizon mixing is not the main reason sequence/graph context failed.

## Best Router By Horizon

| horizon | best router | candidate | all | t50 | t100 diag | hard/failure | easy degradation | switch rate | supported |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | `h10_motion_goal_context` | `motion_goal_context` | 0.069270 | 0.000000 | 0.000000 | 0.072655 | -0.035269 | 0.444293 | True |
| 25 | `h25_baseline_plus_history_goal_neighbor` | `baseline_plus_history_goal_neighbor` | 0.006986 | 0.000000 | 0.000000 | 0.016655 | -0.021896 | 0.104751 | True |
| 50 | `h50_baseline_plus_history_goal_neighbor` | `baseline_plus_history_goal_neighbor` | 0.000001 | 0.000001 | 0.000000 | 0.000001 | -0.000000 | 0.015947 | False |
| 100 | `h100_history_only` | `history_only` | 0.001448 | 0.000000 | 0.001448 | 0.001448 | 0.011562 | 0.128405 | False |

## All Horizon Routers

| key | rows | all | hard/failure | easy | switch | supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `h10_history_only` | 15402 | 0.060896 | 0.059694 | -0.082915 | 0.380535 | True |
| `h25_history_only` | 13470 | -0.000819 | -0.000090 | 0.002556 | 0.145954 | False |
| `h50_history_only` | 11538 | -0.002457 | -0.002457 | 0.013345 | 0.164847 | False |
| `h100_history_only` | 7048 | 0.001448 | 0.001448 | 0.011562 | 0.128405 | False |
| `h10_motion_goal_context` | 15402 | 0.069270 | 0.072655 | -0.035269 | 0.444293 | True |
| `h25_motion_goal_context` | 13470 | -0.000333 | -0.000058 | 0.001295 | 0.110468 | False |
| `h50_motion_goal_context` | 11538 | -0.001980 | -0.001980 | 0.008696 | 0.098371 | False |
| `h100_motion_goal_context` | 7048 | 0.001432 | 0.001432 | 0.009937 | 0.101873 | False |
| `h10_baseline_plus_history_goal_neighbor` | 15402 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | False |
| `h25_baseline_plus_history_goal_neighbor` | 13470 | 0.006986 | 0.016655 | -0.021896 | 0.104751 | True |
| `h50_baseline_plus_history_goal_neighbor` | 11538 | 0.000001 | 0.000001 | -0.000000 | 0.015947 | False |
| `h100_baseline_plus_history_goal_neighbor` | 7048 | 0.001279 | 0.001279 | -0.000398 | 0.187287 | False |

## Sequence / Graph Schema

- sequence_summary_stats: `{'source': 'fresh_run', 'rows': 337991, 'feature_count': 11, 'valid_history_mean': 16.604490280151367, 'valid_history_min': 3.0, 'valid_history_max': 64.0, 'uses_future_endpoint': False, 'uses_future_waypoint': False}`
- graph_summary_stats: `{'source': 'fresh_run', 'rows': 337991, 'frame_groups': 11224, 'rows_with_neighbors': 334525, 'unique_agent_nodes_with_neighbors': 104125, 'max_unique_agents_per_frame': 65, 'feature_count': 34, 'uses_future_endpoint': False, 'uses_future_waypoint': False}`

## Gate

| gate | pass |
| --- | ---: |
| `source_level_split_used` | True |
| `baseline_family_control_loaded` | True |
| `sequence_summary_built` | True |
| `graph_summary_built` | True |
| `horizon_routers_complete` | True |
| `all_horizons_have_test_rows` | True |
| `validation_only_selection` | True |
| `horizon_increment_measured` | True |
| `negative_or_positive_claim_bounded` | True |
| `no_leakage_pass` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |

## Interpretation

- Stage42-IO is a fresh horizon-specific follow-up to Stage42-EQ.
- It tests whether horizon mixing caused the earlier negative sequence/graph context result.
- Positive routers are narrow horizon-specific routing evidence only; negative results keep sequence/graph context diagnostic under this protocol.
- Claims remain raw-frame / dataset-local 2.5D only; no metric/seconds claim, Stage5C, or SMC.
