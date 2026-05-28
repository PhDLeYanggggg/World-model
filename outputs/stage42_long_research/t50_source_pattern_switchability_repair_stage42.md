# Stage42-IR t50 Source-Pattern Switchability Repair

- source: `fresh_stage42_t50_source_pattern_switchability_repair`
- generated_at_utc: `2026-05-28T03:26:39.230674+00:00`
- git_commit: `5cba11c`
- input_hash: `e8b7f59a3e8104475843a9a8001ab0419c498e6d57b4b21748349ee471ff4124`
- gate: `11 / 11`
- verdict: `stage42_ir_t50_source_pattern_switchability_repair_pass`
- repair_supported: `False`
- repair_verdict: `t50_source_pattern_switchability_repair_not_supported`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-IR 是 Stage42-IQ 后续 repair：把 source-pattern support 加到 t50 sequence+graph switchability router。
- source pattern 来自已知 source_file 路径模式，不来自 test endpoint、future waypoint 或 metric calibration。
- future waypoints / endpoints 只作为 train/val supervised labels 或 test evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 仍是 raw-frame horizon，不能写成 seconds-level。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Best Trial

- best_trial_key: `history_only__gain_only`
- test t50 improvement: `0.000000`
- test hard/failure improvement: `0.000000`
- test easy degradation: `-0.000000`
- test switch rate: `0.000000`

## Trial Table

| trial | val t50 | test t50 | hard/failure | easy deg | switch | supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_plus_history_goal_neighbor__gain_harm_guard` | -0.000506 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | False |
| `baseline_plus_history_goal_neighbor__gain_only` | -0.001018 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | False |
| `baseline_plus_history_goal_neighbor__positive_harm_balance` | -0.002252 | 0.000000 | 0.000000 | -0.000000 | 0.000087 | False |
| `history_only__gain_harm_guard` | 0.000648 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | False |
| `history_only__gain_only` | 0.000648 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | False |
| `history_only__positive_harm_balance` | -0.001092 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | False |
| `motion_goal_context__gain_harm_guard` | 0.002499 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | False |
| `motion_goal_context__gain_only` | 0.002499 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | False |
| `motion_goal_context__positive_harm_balance` | 0.001065 | 0.000000 | 0.000000 | -0.000000 | 0.000000 | False |

## Interpretation

Stage42-IR changes source support rather than only thresholds. If it remains unsupported, the context t50 route should be treated as closed under this candidate family; future repair needs new candidate policies or new source data rather than more source-pattern gating.

- This is a source-pattern repair attempt, not a new deployable model unless `repair_supported` is true.
- Claims remain raw-frame / dataset-local 2.5D only; no metric/seconds claim, Stage5C, or SMC.

## Gate

| gate | pass |
| --- | ---: |
| `stage42_iq_loaded` | True |
| `source_pattern_schema_built` | True |
| `t50_rows_present` | True |
| `all_pattern_trials_evaluated` | True |
| `validation_only_selection` | True |
| `test_result_reported` | True |
| `success_or_honest_failure_reported` | True |
| `no_leakage_pass` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
