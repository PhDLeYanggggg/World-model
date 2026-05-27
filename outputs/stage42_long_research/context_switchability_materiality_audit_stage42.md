# Stage42-EE Context Switchability Materiality Audit

- source: `fresh_rerun_stage42_dc_context_switchability_materiality`
- generated_at_utc: `2026-05-27T01:53:35.262802+00:00`
- git_commit: `3de43c8`
- input_hash: `6007b636bb3f47cf9ed3e21ae61404a3f071a452e6c06d38be69bb3545a82bca`
- gate: `12 / 12`
- verdict: `stage42_ee_context_switchability_materiality_audit_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-EE fresh-runs Stage42-DC context switchability and applies a materiality gate.
- 目标是防止把微小 context 增量包装成 scene/goal/neighbor/interaction 主贡献。
- future endpoints / waypoints 只作为 supervised/evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Materiality Summary

- material_delta_threshold: `0.01`
- selected_candidate: `baseline_plus_knn_graph`
- selected_delta_all/t50/hard/easy: `0.000368` / `-0.000074` / `0.000424` / `-0.002388`
- material_context_contribution: `False`
- decision: `context_switchability_materiality_blocked`
- root_cause: Gain/harm context switchability can slightly adjust the baseline-family policy, but the best validation-selected test deltas are far below the 1 percentage-point materiality threshold and t50 is not improved.

## Candidate Deltas Vs Baseline-Family Control

| candidate | all | t50 | hard | easy | delta all | delta t50 | delta hard | delta easy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_plus_knn_graph` | 0.288141 | 0.315351 | 0.276236 | -0.326574 | 0.000368 | -0.000074 | 0.000424 | -0.002388 |
| `baseline_plus_graph_history_scalar` | 0.288071 | 0.315348 | 0.276155 | -0.326212 | 0.000298 | -0.000078 | 0.000343 | -0.002026 |
| `baseline_plus_scalar_neighbor` | 0.287773 | 0.315425 | 0.275812 | -0.324186 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `baseline_plus_graph_goal` | 0.287691 | 0.315264 | 0.275720 | -0.323822 | -0.000082 | -0.000162 | -0.000092 | 0.000364 |
| `baseline_plus_goal_scene` | 0.287471 | 0.315182 | 0.275473 | -0.323767 | -0.000302 | -0.000243 | -0.000339 | 0.000419 |

## Interpretation

- Stage42-EE is a fresh rerun/materiality audit of the gain-harm context target, not a new data conversion or Stage5C run.
- The current context switchability route remains useful as a negative result: it prevents overclaiming tiny graph/goal/neighbor increments as a main contribution.
- The next context attempt must change target/data support rather than repeat residual or current gain-harm switchability.

## Gate

| gate | pass |
| --- | ---: |
| `dc_fresh_rerun_passed` | True |
| `multiple_context_candidates_checked` | True |
| `materiality_threshold_recorded` | True |
| `selected_delta_recorded` | True |
| `context_materiality_decision_recorded` | True |
| `micro_delta_not_overclaimed` | True |
| `root_cause_written` | True |
| `next_action_written` | True |
| `no_future_or_test_leakage` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
