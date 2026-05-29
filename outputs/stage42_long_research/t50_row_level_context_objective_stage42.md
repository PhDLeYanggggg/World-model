# Stage42-KB t50 Row-Level Context Objective

- source: `fresh_stage42_kb_t50_row_level_context_objective`
- generated_at_utc: `2026-05-29T08:54:57.587059+00:00`
- git_commit: `bd9e19c`
- input_hash: `e0cd67ab94778cf36df534fef2a5aefcba5f4addfd7408b81a11c992029bfd0d`
- gate: `12 / 12`
- verdict: `stage42_kb_t50_row_level_context_objective_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-KB 是 KA 后续 t50 row-level source/horizon objective fresh experiment，不是 metric 或 seconds-level 结果。
- 本实验训练 expected-gain switcher 来决定何时从 baseline-family protected control 切到 context proposal。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Best Validation-Selected Trial

- candidate: `baseline_plus_history`
- feature_set: `context_only`
- min_abs_gain_margin: `0.0`
- t50 improvement vs baseline-family: `0.00%`
- all/hard/easy: `0.00%` / `0.00%` / `-0.00%`
- oracle t50 headroom for same candidate: `1.20%`
- switch diagnostics: `{'switch_rate_test': 0.0020800832033281333, 'positive_gain_rate_test': 0.13529207834980067, 'switched_positive_rate': 0.0, 'switched_harm_rate': 0.0, 'missed_positive_rate': 0.13557408372416188, 'capture_rate': 0.0, 'mean_pred_gain_switched': 0.0587780004118106}`
- deployable_increment_supported: `False`
- failure_or_success_reason: `validation_safe_policy_under_switches`

## Top Trials By t50

| candidate | feature_set | margin | t50 | all | hard | easy | switch | capture |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_plus_history` | `context_only` | 0.000 | 0.00% | 0.00% | 0.00% | -0.00% | 0.21% | 0.00% |
| `baseline_plus_history` | `context_only` | 0.010 | 0.00% | 0.00% | 0.00% | -0.00% | 6.27% | 0.00% |
| `baseline_plus_history` | `context_only` | 0.050 | 0.00% | 0.00% | 0.00% | -0.00% | 9.40% | 0.00% |
| `baseline_plus_history` | `context_plus_baseline_family` | 0.000 | 0.00% | 0.00% | 0.00% | -0.00% | 0.03% | 0.00% |
| `baseline_plus_history` | `context_plus_baseline_family` | 0.010 | 0.00% | 0.00% | 0.00% | -0.00% | 3.67% | 0.00% |
| `baseline_plus_history` | `context_plus_baseline_family` | 0.050 | 0.00% | 0.00% | 0.00% | -0.00% | 8.36% | 0.00% |
| `baseline_plus_history` | `baseline_family_only` | 0.000 | 0.00% | 0.00% | 0.00% | -0.00% | 0.10% | 0.00% |
| `baseline_plus_history` | `baseline_family_only` | 0.010 | 0.00% | 0.00% | 0.00% | -0.00% | 11.65% | 0.00% |
| `baseline_plus_history` | `baseline_family_only` | 0.050 | 0.00% | 0.00% | 0.00% | -0.00% | 1.99% | 0.00% |
| `baseline_plus_history` | `all_source_features` | 0.000 | 0.00% | 0.00% | 0.00% | -0.00% | 0.03% | 0.00% |
| `baseline_plus_history` | `all_source_features` | 0.010 | 0.00% | 0.00% | 0.00% | -0.00% | 3.67% | 0.00% |
| `baseline_plus_history` | `all_source_features` | 0.050 | 0.00% | 0.00% | 0.00% | -0.00% | 8.36% | 0.00% |

## Bootstrap

| slice | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `t50` | 0.00% | 0.00% | 0.00% | 11538 |
| `hard_failure_t50` | 0.00% | 0.00% | 0.00% | 11538 |
| `easy_t50_degradation` | 0.00% | 0.00% | 0.00% | 3346 |

## Gate

| gate | pass |
| --- | ---: |
| `ka_contract_loaded` | `True` |
| `t50_row_level_trials_complete` | `True` |
| `validation_only_selection` | `True` |
| `t50_test_rows_present` | `True` |
| `oracle_headroom_measured` | `True` |
| `deployable_increment_supported_or_failure_reason_recorded` | `True` |
| `easy_safety_enforced` | `True` |
| `bootstrap_available` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_3d_foundation` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |

## Interpretation

- KB is a fresh t50 row-level objective follow-up to KA.
- It trains expected-gain switching for context proposals over the baseline-family protected control.
- A positive result may support the next t50 source/horizon objective; a negative result preserves KA's blocker rather than hiding it.
