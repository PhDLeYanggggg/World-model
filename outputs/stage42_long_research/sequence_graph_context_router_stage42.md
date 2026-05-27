# Stage42-EQ Sequence+Graph Context Router

- source: `fresh_stage42_sequence_graph_context_router`
- generated_at_utc: `2026-05-27T03:23:27.074561+00:00`
- git_commit: `c0e8255`
- input_hash: `3b2c8dc2606a2a66dd6c7cd36a3c76fc8d900513a1898ec30469e5913a45dc35`
- gate: `12 / 12`
- verdict: `stage42_eq_sequence_graph_context_router_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-EQ 是 source-compatible sequence+graph context router fresh probe，不是新数据转换或 metric/seconds-level 结果。
- 该实验不让 sequence/graph 直接替代 floor，而是用 past-only sequence summary 与 current-frame graph summary 决定 context proposal 是否值得切换。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- router_target: `predict supervised gain of context proposal over baseline-family protected control using sequence+graph past-only summaries`
- candidates: `['history_only', 'motion_goal_context', 'baseline_plus_history_goal_neighbor']`
- positive_sequence_graph_context_routers: `[]`
- best_router: `baseline_plus_history_goal_neighbor`
- sequence_graph_increment_verdict: `stage42_eq_sequence_graph_context_router_not_supported`

Stage42-EQ tests whether sequence/graph context is useful as a safe switchability signal after direct sequence and graph residual prediction failed. Positive routers would support a narrow deployment-router contribution. Negative routers keep context as diagnostic under this protocol.

## Router Results vs Baseline-Family Protected Control

| candidate | features | all | t50 | t100 diag | hard/failure | easy degradation | switch rate | increment supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `history_only` | 173 | -0.001402 | -0.000278 | 0.000000 | -0.001153 | -0.000358 | 0.138059 | False |
| `motion_goal_context` | 183 | -0.000980 | -0.000022 | 0.000000 | -0.000419 | 0.007993 | 0.052130 | False |
| `baseline_plus_history_goal_neighbor` | 211 | 0.000118 | -0.000197 | 0.000083 | 0.000169 | -0.001971 | 0.045956 | False |

## Sequence / Graph Schema

- sequence_summary_stats: `{'source': 'fresh_run', 'rows': 337991, 'feature_count': 11, 'valid_history_mean': 16.604490280151367, 'valid_history_min': 3.0, 'valid_history_max': 64.0, 'uses_future_endpoint': False, 'uses_future_waypoint': False}`
- graph_summary_stats: `{'source': 'fresh_run', 'rows': 337991, 'frame_groups': 11224, 'rows_with_neighbors': 334525, 'unique_agent_nodes_with_neighbors': 104125, 'max_unique_agents_per_frame': 65, 'feature_count': 34, 'uses_future_endpoint': False, 'uses_future_waypoint': False}`

## Interpretation

- This experiment is a stricter follow-up to negative sequence-residual and graph-residual probes.
- It tests sequence/graph context only as a validated router signal over a protected baseline-family control.
- If the result is negative, scene/goal/neighbor/sequence context remains diagnostic under this source-level protocol.
- Claims remain raw-frame / dataset-local 2.5D only; no metric/seconds claim, Stage5C, or SMC.

## Gate

| gate | pass |
| --- | ---: |
| `source_level_split_used` | True |
| `baseline_family_control_loaded` | True |
| `sequence_summary_built` | True |
| `graph_summary_built` | True |
| `router_candidates_complete` | True |
| `validation_only_selection` | True |
| `sequence_graph_increment_measured` | True |
| `negative_or_positive_claim_bounded` | True |
| `no_leakage_pass` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
