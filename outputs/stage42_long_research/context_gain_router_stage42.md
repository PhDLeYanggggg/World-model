# Stage42-EL Context Gain Router

- source: `fresh_stage42_context_gain_router`
- generated_at_utc: `2026-05-27T02:39:40.067047+00:00`
- git_commit: `2df9224`
- input_hash: `aa0ffc5e6a7105a48a4544f56338524924924b5685d73ffcb8b49bb4fc878004`
- gate: `10 / 10`
- verdict: `stage42_el_context_gain_router_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-EL 是 context gain-router fresh probe，不是重复 sequence/graph residual-delta protocol。
- router target 是 baseline-family protected floor 与 context proposal 的 supervised gain/harm，而不是直接预测 residual 轨迹。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- router_target: `predict supervised gain of context proposal over baseline-family protected control`
- candidates: `['history_only', 'motion_goal_context', 'baseline_plus_history_goal_neighbor']`
- positive_context_gain_routers: `[]`
- best_router: `baseline_plus_history_goal_neighbor`
- context_increment_verdict: `stage42_el_context_gain_router_not_supported`

This is a deployment-aligned context target. Positive routers would support context as a safe switchability signal over the baseline-family control. Negative routers preserve the current conclusion that context is not yet an independent deployable contribution under this source-level protocol.

## Router Results vs Baseline-Family Protected Control

| candidate | all | t50 | hard/failure | easy degradation | switch rate | increment supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `history_only` | -0.007183 | -0.002797 | -0.004945 | 0.005166 | 0.201420 | False |
| `motion_goal_context` | -0.004646 | -0.001104 | -0.003411 | 0.010222 | 0.099351 | False |
| `baseline_plus_history_goal_neighbor` | 0.000278 | -0.000019 | 0.000321 | -0.002666 | 0.027393 | False |

## Interpretation

- This stage changes the context target from residual trajectory prediction to gain/harm routing over a strong baseline-family control.
- A negative result keeps scene/goal/neighbor/interaction as diagnostic under this protocol; it does not prove context can never help.
- A positive result would support context as a safe switchability signal only, not as metric/seconds-level or true-3D evidence.

## Gate

| gate | pass |
| --- | ---: |
| `source_level_split_used` | True |
| `baseline_family_control_loaded` | True |
| `gain_router_candidates_complete` | True |
| `validation_only_selection` | True |
| `context_increment_measured` | True |
| `negative_or_positive_claim_bounded` | True |
| `no_leakage_pass` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
