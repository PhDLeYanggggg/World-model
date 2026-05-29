# Stage42-IQ t50 Switchability Calibration Repair

- source: `fresh_stage42_t50_switchability_calibration_repair`
- generated_at_utc: `2026-05-29T04:42:56.365448+00:00`
- git_commit: `0ca07d9`
- input_hash: `7c51c78a99a22c0c858d55eeae5c662efd484f76c71a5a305474bf60a074bc28`
- gate: `11 / 11`
- verdict: `stage42_iq_t50_switchability_calibration_repair_pass`
- repair_supported: `False`
- repair_verdict: `validation_selected_gain_harm_router_still_fails_to_capture_t50_headroom`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-IQ 是 Stage42-IP 后续 repair：只修 t50 sequence+graph under-switching，不生成新 metric/seconds-level 结果。
- gain / harm / positive-gain targets 只在 train/val 监督中使用 future labels；inference features 仍是 past-only sequence+graph + causal source-level features。
- test set 只最终评估一次，不用于阈值选择。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 future endpoint 作为输入。
- t+50 仍是 raw-frame horizon，不能写成 seconds-level。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Best Trial

- best_trial_key: `baseline_plus_history_goal_neighbor__gain_only`
- test t50 improvement: `0.000001`
- test hard/failure improvement: `0.000001`
- test easy degradation: `-0.000000`
- test switch rate: `0.005720`
- validation t50 improvement: `-0.001745`

## Trial Table

| trial | val t50 | test t50 | hard/failure | easy deg | switch | supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_plus_history_goal_neighbor__gain_harm_guard` | 0.001437 | 0.000001 | 0.000001 | -0.000000 | 0.008407 | False |
| `baseline_plus_history_goal_neighbor__gain_only` | -0.001745 | 0.000001 | 0.000001 | -0.000000 | 0.005720 | False |
| `baseline_plus_history_goal_neighbor__positive_harm_balance` | 0.001274 | 0.000001 | 0.000001 | -0.000000 | 0.003293 | False |
| `history_only__gain_harm_guard` | -0.001211 | -0.001241 | -0.001241 | 0.005435 | 0.082423 | False |
| `history_only__gain_only` | -0.002477 | -0.001094 | -0.001094 | 0.005547 | 0.085890 | False |
| `history_only__positive_harm_balance` | -0.001325 | -0.000496 | -0.000496 | 0.002973 | 0.061882 | False |
| `motion_goal_context__gain_harm_guard` | -0.000092 | -0.000181 | -0.000181 | 0.001440 | 0.021754 | False |
| `motion_goal_context__gain_only` | 0.000221 | -0.000341 | -0.000341 | 0.001489 | 0.025394 | False |
| `motion_goal_context__positive_harm_balance` | 0.000016 | -0.000650 | -0.000650 | 0.001895 | 0.036921 | False |

## Interpretation

Stage42-IQ formally tests whether t50 under-switching can be repaired by supervised gain/harm calibration. If the best validation-selected policy still fails on test, the next repair should not be more threshold tuning; it should change supervision, source support, or candidate policy families.

- This is a repair attempt, not a new deployable model unless `repair_supported` is true.
- Claims remain raw-frame / dataset-local 2.5D only; no metric/seconds claim, Stage5C, or SMC.

## Gate

| gate | pass |
| --- | ---: |
| `stage42_ip_blocker_loaded` | True |
| `t50_rows_present` | True |
| `all_candidates_evaluated` | True |
| `gain_harm_targets_trained` | True |
| `validation_only_selection` | True |
| `test_result_reported` | True |
| `repair_success_or_honest_failure_reported` | True |
| `no_leakage_pass` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
