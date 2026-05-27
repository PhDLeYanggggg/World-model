# Stage42-GK Context Switchability Family Audit

- source: `fresh_stage42_gk_context_switchability_family_audit`
- generated_at_utc: `2026-05-27T12:48:09.065350+00:00`
- git_commit: `f16cf57`
- gate: `14 / 14`
- verdict: `stage42_gk_context_switchability_family_audit_pass`
- decision: `context_switchability_family_not_supported`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-GK 是 changed-target context switchability/gain/harm family audit，不重复已关闭的 residual sequence/graph protocol。
- router target 是 feature-family proposal 相对 baseline-family control 的 gain/harm/switchability，而不是直接 residual trajectory target。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- material_delta_threshold: `0.01`
- best_family: `baseline_plus_history_goal_neighbor`
- material_context_families: `[]`
- material_context_contribution_supported: `False`
- root_cause: Feature-family context gain/harm targets still do not yield a material positive, easy-safe lift over baseline-family control.
- next_action: Do not claim scene/goal/neighbor as independent main contribution if material families remain empty; next attempt must change source support or full-sequence target.

## Family Rows

| family | features | all | t50 | t100 raw | hard | easy | switch | material |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `history_only` | 128 | -0.007230 | -0.002657 | 0.000000 | -0.004998 | 0.005166 | 0.200598 | False |
| `goal_only` | 17 | -0.031306 | -0.006128 | -0.000341 | -0.013766 | 0.071891 | 0.190442 | False |
| `neighbor_only` | 12 | -0.000232 | -0.000149 | 0.000000 | -0.000085 | 0.001355 | 0.013507 | False |
| `motion_goal_context` | 138 | -0.004646 | -0.001104 | 0.000000 | -0.003411 | 0.010222 | 0.099351 | False |
| `baseline_plus_history` | 156 | 0.000001 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.012622 | False |
| `baseline_plus_goal` | 45 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.015846 | False |
| `baseline_plus_neighbor` | 40 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0.045029 | False |
| `baseline_plus_history_goal_neighbor` | 166 | -0.000003 | 0.000000 | 0.000000 | 0.000006 | 0.000093 | 0.031733 | False |

## Interpretation

- This is a changed-target context audit: gain/harm/switchability is the trained target, not residual trajectory deltas.
- If material_context_families is empty, scene/goal/neighbor/history context remains blocked as an independent main contribution under this target.
- A negative result still matters: it narrows the next credible experiment to source support or genuinely different full-sequence/group objectives.

## Gate

| gate | pass |
| --- | ---: |
| `source_fresh` | True |
| `baseline_family_control_loaded` | True |
| `changed_target_gain_harm_used` | True |
| `feature_families_checked` | True |
| `validation_selection_test_once_recorded` | True |
| `materiality_decision_recorded` | True |
| `claim_boundary_matches_materiality` | True |
| `root_cause_written` | True |
| `next_action_written` | True |
| `no_future_or_test_leakage` | True |
| `no_metric_seconds_overclaim` | True |
| `not_true3d_or_foundation` | True |
| `stage5c_false` | True |
| `smc_false` | True |
