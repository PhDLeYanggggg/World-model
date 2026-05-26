# Stage42-DP Context Model Closure

- source: `fresh_synthesis_after_fresh_ar_as_rerun`
- generated_at_utc: `2026-05-26T23:47:02.993401+00:00`
- git_commit: `950aa86`
- gate: `19 / 19`
- verdict: `stage42_dp_context_model_closure_pass`
- closure_decision: `close_current_sequence_graph_residual_context_protocol`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DP 是 source-level sequence/graph context closure，不重新训练新模型，不调 test threshold。
- 本阶段整合 fresh Stage42-AR sequence-context 和 fresh Stage42-AS graph-context rerun。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Baseline-Family Control

- rows: `47458.0`
- all_improvement: `0.2877734037648393`
- t50_improvement: `0.31542535139554606`
- t100_raw_frame_diagnostic_improvement: `0.14282475620015533`
- hard_failure_improvement: `0.2758122379367457`
- easy_degradation: `-0.32418582524688455`

## Sequence / Graph Deltas vs Baseline-Family Control

| protocol | variant | delta all | delta t50 | delta hard/failure | delta easy |
| --- | --- | ---: | ---: | ---: | ---: |
| `sequence_context` | `sequence_history` | -0.024458 | -0.083057 | -0.028398 | 0.014358 |
| `sequence_context` | `sequence_goal_neighbor_no_history` | -0.026473 | -0.092136 | -0.029072 | 0.038166 |
| `sequence_context` | `sequence_history_goal_neighbor` | -0.087523 | -0.087529 | -0.096088 | 0.053745 |
| `graph_context` | `graph_only` | -0.023503 | -0.085752 | -0.027002 | 0.006648 |
| `graph_context` | `graph_goal` | -0.023123 | -0.086390 | -0.026341 | 0.007873 |
| `graph_context` | `graph_history_goal` | -0.023009 | -0.086417 | -0.026235 | 0.007797 |

## Closure Decision

- positive_context_rows: `[]`
- best_delta_all: `-0.0230085146797967`
- best_delta_t50: `-0.08305738715518596`
- best_delta_hard_failure: `-0.026234695463867364`
- worst_delta_all: `-0.08752342984606587`
- worst_delta_t50: `-0.09213611361912277`
- prior_context_rescue_decision: `stop_repeating_current_context_residual_or_gated_protocols`
- prior_context_switchability_decision: `context_switchability_not_supported`

Fresh Stage42-AR/AS reruns show that temporal sequence context and current-frame kNN graph context both reduce all/t50/hard-failure improvements relative to the baseline-family first-stage control. The dominant current signal remains baseline-family rollout context plus safety floor; the present residual context target is not extracting independent scene/goal/interaction value.

## Next Best Action

- Do not repeat the same residual sequence/graph context protocol without changing target or data support.
- Prioritize source/legal/time closure for ETH_UCY, TrajNet, and UCY so context can be tested on better-calibrated sources.
- If context modeling is revisited, use switchability/gain-harm or full sequence architecture with baseline-family control, not blind residual deltas.
- Keep Stage37/teacher floor and Stage42 protected runtime policies as deployable evidence while context remains auxiliary/diagnostic.

## Claim Boundary

- This is fresh evidence for closing the current sequence/graph residual context protocol, not evidence that context can never help.
- It does not execute Stage5C and does not enable SMC.
- It remains protected dataset-local/raw-frame 2.5D evidence, not metric/seconds-level, true 3D, or foundation evidence.
