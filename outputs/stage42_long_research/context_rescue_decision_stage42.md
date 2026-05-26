# Stage42-DB Context Rescue Decision Audit

- source: `fresh_synthesis_from_cached_verified_context_runs`
- generated_at_utc: `2026-05-26T20:55:45.287348+00:00`
- git_commit: `cc4f4d7`
- gate: `13 / 13`
- verdict: `stage42_db_context_rescue_decision_pass`
- decision: `stop_repeating_current_context_residual_or_gated_protocols`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-DB 是 context rescue decision audit，不重新训练，不调 threshold，不把 cached 结果写成 fresh training。
- 本阶段整合 CJ/CK/AR/AS 已训练或已评估的 context evidence，判断是否应继续同类 protocol。
- future endpoints / waypoints 只作为 labels/eval，不作为 inference input。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Protocol Status

| protocol | loaded | source | verdict |
| --- | ---: | --- | --- |
| `goal_scene_gated` | `True` | `fresh_run` | `stage42_cj_goal_scene_gated_expert_pass_diagnostic_no_overclaim` |
| `neighbor_interaction_gated` | `True` | `fresh_run` | `stage42_ck_neighbor_interaction_gated_expert_pass_diagnostic_no_overclaim` |
| `sequence_context` | `True` | `fresh_run` | `stage42_ar_sequence_context_evidence_partial_or_negative` |
| `graph_context` | `True` | `fresh_run` | `stage42_as_graph_context_evidence_partial_or_negative` |

## Context Variant Deltas vs Baseline-Family Control

| protocol | variant | delta all | delta t50 | delta hard/failure | delta easy | safe positive? |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `goal_scene_gated` | `goal_scene_only_control` | -0.287773 | -0.315425 | -0.275812 | 0.324186 | `False` |
| `goal_scene_gated` | `baseline_plus_goal_scene` | -0.025271 | -0.087785 | -0.027164 | 0.022474 | `False` |
| `goal_scene_gated` | `baseline_plus_motion_goal_context` | -0.041985 | -0.095255 | -0.038318 | 0.067559 | `False` |
| `neighbor_interaction_gated` | `baseline_plus_scalar_neighbor` | -0.024026 | -0.085810 | -0.027021 | 0.010997 | `False` |
| `neighbor_interaction_gated` | `baseline_plus_goal_scene` | -0.025271 | -0.087785 | -0.027164 | 0.022474 | `False` |
| `neighbor_interaction_gated` | `baseline_plus_history_scalar` | -0.024437 | -0.085935 | -0.027431 | 0.011222 | `False` |
| `neighbor_interaction_gated` | `baseline_plus_knn_graph` | -0.043950 | -0.091653 | -0.038039 | 0.067763 | `False` |
| `neighbor_interaction_gated` | `baseline_plus_graph_goal` | -0.081026 | -0.093372 | -0.087710 | 0.051921 | `False` |
| `neighbor_interaction_gated` | `baseline_plus_graph_history_scalar` | -0.092072 | -0.091456 | -0.091209 | 0.080034 | `False` |
| `sequence_context` | `sequence_history` | -0.024458 | -0.083057 | -0.028398 | 0.014358 | `False` |
| `sequence_context` | `sequence_goal_neighbor_no_history` | -0.026473 | -0.092136 | -0.029072 | 0.038166 | `False` |
| `sequence_context` | `sequence_history_goal_neighbor` | -0.087523 | -0.087529 | -0.096088 | 0.053745 | `False` |
| `graph_context` | `graph_only` | -0.023503 | -0.085752 | -0.027002 | 0.006648 | `False` |
| `graph_context` | `graph_goal` | -0.023123 | -0.086390 | -0.026341 | 0.007873 | `False` |
| `graph_context` | `graph_history_goal` | -0.023009 | -0.086417 | -0.026235 | 0.007797 | `False` |

## Decision

- safe_positive_context_variants: `[]`
- best_delta_all: `-0.0230085146797967`
- best_delta_t50: `-0.08305738715518596`
- best_delta_hard_failure: `-0.026234695463867364`
- root_cause: Existing goal/scene, neighbor/interaction, sequence, and graph context variants either reduce all/t50/hard or add easy risk after baseline-family rollout context. The next credible experiment must change the target/model/data, not merely rerun the same residual/gated variants or tune thresholds.

## Required Next Protocol Change

- Use source-compatible graph/sequence model with a different supervision target, such as switchability/gain-harm labels rather than residual waypoint delta only.
- Add legal/source-calibrated data where scene/goal/interaction context can vary independently from baseline-family rollout context.
- Keep baseline-family rollout as the control arm and require validation-only safety gates plus bootstrap-positive test evidence.

## Interpretation

- Stage42-DB does not say scene/goal/neighbor/history can never help.
- It says the current residual/gated protocols are exhausted under available source-level evidence.
- Future work must change model target/model family/data support, while retaining baseline-family control and validation-only safety gates.
- Claims remain protected dataset-local/raw-frame 2.5D, not true 3D, not foundation, not metric/seconds-level, not Stage5C, and not SMC.
