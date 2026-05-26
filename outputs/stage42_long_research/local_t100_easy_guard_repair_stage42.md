# Stage42-BI Local T100 Source-Robust Easy Guard Repair

- source: `fresh_source_robust_easy_guard_repair`
- generated_at_utc: `2026-05-26T13:06:27.415034+00:00`
- git_commit: `10a7491`
- input_hash: `a901cc78ec389badfe0f16384ce25170de1ae9e8de0eb28042a08962c60c9ee3`
- gate: `14 / 14`
- verdict: `stage42_bi_ucy_t100_easy_guard_repair_pass_with_global_blocker`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BI 修复 Stage42-BH 暴露的 UCY independent-source t100 easy degradation。
- 策略选择只使用 non-holdout source：validation source + train sources；holdout source 只评估一次。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t100 仍是 raw-frame diagnostic；UCY 修复不等于全局 t100 success。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Repair Strategy

- description: Candidate policies must be positive and easy-safe on every non-holdout source (validation source plus train sources), then are evaluated once on the held-out independent source.
- threshold_quantiles: `[0.0, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.975, 0.99]`
- holdout_used_for_selection: `False`

## Summary

- independent_t100_sources: `5`
- ucy_independent_sources: `4`
- eth_ucy_independent_sources: `1`
- trajnet_independent_sources: `0`
- ucy_t100_source_cv_supported: `True`
- ucy_t100_mean_improvement_vs_fallback: `0.44591415479775254`
- ucy_t100_min_improvement_vs_fallback: `0.42531297412698216`
- ucy_t100_max_easy_degradation: `0.011339719285930428`
- bh_previous_ucy_max_easy_degradation: `0.06332289296349253`
- blocked_domains: `ETH_UCY, TrajNet`
- global_t100_positive_claim_allowed: `False`

## Domain Source-CV Summary

| domain | horizon | folds | safe folds | mean improvement | min improvement | max easy degradation | all safe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `UCY` | 50 | 4 | 0 | 0.090350 | 0.000000 | 0.185291 | False |
| `UCY` | 100 | 4 | 4 | 0.445914 | 0.425313 | 0.011340 | True |

## Fold Details

| domain | holdout | horizon | selected policy | rows | improvement | easy degradation | switch rate |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `UCY` | `UCY/students03/obsmat_px.txt` | 50 | `robust_speed_damped_velocity_0p50_gte_0.0239733` | 6252 | 0.361399 | 0.185291 | 0.481926 |
| `UCY` | `UCY/students03/obsmat_px.txt` | 100 | `robust_speed_damped_velocity_0p50_gte_0.00175007` | 3342 | 0.441619 | 0.011340 | 0.352184 |
| `UCY` | `UCY/zara02/obsmat.txt` | 50 | `global_constant_velocity_causal_fd` | 2787 | 0.000000 | 0.000000 | 0.000000 |
| `UCY` | `UCY/zara02/obsmat.txt` | 100 | `robust_speed_damped_velocity_0p50_gte_0.00820814` | 2071 | 0.474108 | 0.000000 | 0.191695 |
| `UCY` | `UCY/students01/students001.txt` | 50 | `global_constant_velocity_causal_fd` | 6134 | 0.000000 | 0.000000 | 0.000000 |
| `UCY` | `UCY/students01/students001.txt` | 100 | `robust_speed_damped_velocity_0p50_gte_0.00170823` | 1866 | 0.442617 | -0.076291 | 1.000000 |
| `UCY` | `UCY/zara01/obsmat.txt` | 50 | `global_constant_velocity_causal_fd` | 213 | 0.000000 | 0.000000 | 0.000000 |
| `UCY` | `UCY/zara01/obsmat.txt` | 100 | `robust_speed_damped_velocity_0p50_gte_0.0102316` | 95 | 0.425313 | -0.184084 | 0.589474 |

## Interpretation

- Stage42-BI repairs the Stage42-BH UCY independent-source t100 easy-degradation blocker using only non-holdout sources for selection.
- The repair keeps all UCY t100 independent-source folds positive and easy-safe.
- This is still not a global t100 claim because ETH_UCY and TrajNet do not have enough independent t100 sources.
- No metric/seconds-level, true-3D, Stage5C, or SMC claim is allowed.
