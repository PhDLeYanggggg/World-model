# Stage42-EY Continuous Group-Risk Repair

- source: `fresh_stage42_continuous_group_risk_repair`
- generated_at_utc: `2026-05-27T05:20:26.797966+00:00`
- git_commit: `225c360`
- input_hash: `da1c2d3589a89ab3258e92f4f64eaba7bd1ede88516191210b9fabb67354aade`
- gate: `16 / 18`
- verdict: `stage42_ey_continuous_group_risk_repair_positive_not_promoted`
- decision: `stage42_ey_continuous_group_risk_repair_positive_not_promoted`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-EX 证明 binary group-risk 几乎饱和，无法提供比 Stage42-DI 更强的 adaptive repair 信号。
- Stage42-EY 将 group risk 改为 continuous predicted-geometry risk score，再用 validation-only quantile buckets 冻结 repair rules。
- risk score 只使用 predicted/base/floor rollout geometry、source/frame/horizon group key、agent id、当前/过去可得信息。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- risk bucket 阈值、mode、slice rule、candidate 只在 validation 上选择；test 只按冻结规则执行。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Selected Repair

- mode: `domain_horizon`
- validation_score: `1.893497`
- test_score: `0.978820`
- thresholds: `[]`
- bucket_counts_val: `{'0': 23788}`
- bucket_counts_test: `{'0': 47458}`
- candidate_usage: `{'c00|fallback_unsafe|sep0.02|margin0.0': 0, 'c01|predicted_safe_only|sep0.02|margin0.0|safe_min_sep0.02': 0, 'c02|blend_unsafe|sep0.02|margin0.0|alpha0.25': 0, 'c03|blend_unsafe|sep0.02|margin0.0|alpha0.5': 0, 'c04|blend_unsafe|sep0.02|margin0.0|alpha0.75': 0, 'c05|repel_unsafe|sep0.02|margin0.0|strength0.25': 0, 'c06|repel_unsafe|sep0.02|margin0.0|strength0.5': 0, 'c07|fallback_unsafe|sep0.02|margin0.005': 0, 'c08|predicted_safe_only|sep0.02|margin0.005|safe_min_sep0.02': 0, 'c09|blend_unsafe|sep0.02|margin0.005|alpha0.25': 0, 'c10|blend_unsafe|sep0.02|margin0.005|alpha0.5': 0, 'c11|blend_unsafe|sep0.02|margin0.005|alpha0.75': 0, 'c12|repel_unsafe|sep0.02|margin0.005|strength0.25': 0, 'c13|repel_unsafe|sep0.02|margin0.005|strength0.5': 0, 'c14|fallback_unsafe|sep0.02|margin0.01': 0, 'c15|predicted_safe_only|sep0.02|margin0.01|safe_min_sep0.02': 0, 'c16|blend_unsafe|sep0.02|margin0.01|alpha0.25': 0, 'c17|blend_unsafe|sep0.02|margin0.01|alpha0.5': 0, 'c18|blend_unsafe|sep0.02|margin0.01|alpha0.75': 0, 'c19|repel_unsafe|sep0.02|margin0.01|strength0.25': 0, 'c20|repel_unsafe|sep0.02|margin0.01|strength0.5': 0, 'c21|fallback_unsafe|sep0.05|margin0.0': 0, 'c22|predicted_safe_only|sep0.05|margin0.0|safe_min_sep0.05': 0, 'c23|blend_unsafe|sep0.05|margin0.0|alpha0.25': 0, 'c24|blend_unsafe|sep0.05|margin0.0|alpha0.5': 0, 'c25|blend_unsafe|sep0.05|margin0.0|alpha0.75': 0, 'c26|repel_unsafe|sep0.05|margin0.0|strength0.25': 0, 'c27|repel_unsafe|sep0.05|margin0.0|strength0.5': 0, 'c28|fallback_unsafe|sep0.05|margin0.005': 0, 'c29|predicted_safe_only|sep0.05|margin0.005|safe_min_sep0.05': 0, 'c30|blend_unsafe|sep0.05|margin0.005|alpha0.25': 0, 'c31|blend_unsafe|sep0.05|margin0.005|alpha0.5': 0, 'c32|blend_unsafe|sep0.05|margin0.005|alpha0.75': 0, 'c33|repel_unsafe|sep0.05|margin0.005|strength0.25': 0, 'c34|repel_unsafe|sep0.05|margin0.005|strength0.5': 0, 'c35|fallback_unsafe|sep0.05|margin0.01': 0, 'c36|predicted_safe_only|sep0.05|margin0.01|safe_min_sep0.05': 0, 'c37|blend_unsafe|sep0.05|margin0.01|alpha0.25': 0, 'c38|blend_unsafe|sep0.05|margin0.01|alpha0.5': 0, 'c39|blend_unsafe|sep0.05|margin0.01|alpha0.75': 0, 'c40|repel_unsafe|sep0.05|margin0.01|strength0.25': 0, 'c41|repel_unsafe|sep0.05|margin0.01|strength0.5': 0, 'c42|fallback_unsafe|sep0.08|margin0.0': 0, 'c43|predicted_safe_only|sep0.08|margin0.0|safe_min_sep0.08': 0, 'c44|blend_unsafe|sep0.08|margin0.0|alpha0.25': 0, 'c45|blend_unsafe|sep0.08|margin0.0|alpha0.5': 0, 'c46|blend_unsafe|sep0.08|margin0.0|alpha0.75': 0, 'c47|repel_unsafe|sep0.08|margin0.0|strength0.25': 10770, 'c48|repel_unsafe|sep0.08|margin0.0|strength0.5': 36688, 'c49|fallback_unsafe|sep0.08|margin0.005': 0, 'c50|predicted_safe_only|sep0.08|margin0.005|safe_min_sep0.08': 0, 'c51|blend_unsafe|sep0.08|margin0.005|alpha0.25': 0, 'c52|blend_unsafe|sep0.08|margin0.005|alpha0.5': 0, 'c53|blend_unsafe|sep0.08|margin0.005|alpha0.75': 0, 'c54|repel_unsafe|sep0.08|margin0.005|strength0.25': 0, 'c55|repel_unsafe|sep0.08|margin0.005|strength0.5': 0, 'c56|fallback_unsafe|sep0.08|margin0.01': 0, 'c57|predicted_safe_only|sep0.08|margin0.01|safe_min_sep0.08': 0, 'c58|blend_unsafe|sep0.08|margin0.01|alpha0.25': 0, 'c59|blend_unsafe|sep0.08|margin0.01|alpha0.5': 0, 'c60|blend_unsafe|sep0.08|margin0.01|alpha0.75': 0, 'c61|repel_unsafe|sep0.08|margin0.01|strength0.25': 0, 'c62|repel_unsafe|sep0.08|margin0.01|strength0.5': 0, 'c63|fallback_unsafe|sep0.12|margin0.0': 0, 'c64|predicted_safe_only|sep0.12|margin0.0|safe_min_sep0.12': 0, 'c65|blend_unsafe|sep0.12|margin0.0|alpha0.25': 0, 'c66|blend_unsafe|sep0.12|margin0.0|alpha0.5': 0, 'c67|blend_unsafe|sep0.12|margin0.0|alpha0.75': 0, 'c68|repel_unsafe|sep0.12|margin0.0|strength0.25': 0, 'c69|repel_unsafe|sep0.12|margin0.0|strength0.5': 0, 'c70|fallback_unsafe|sep0.12|margin0.005': 0, 'c71|predicted_safe_only|sep0.12|margin0.005|safe_min_sep0.12': 0, 'c72|blend_unsafe|sep0.12|margin0.005|alpha0.25': 0, 'c73|blend_unsafe|sep0.12|margin0.005|alpha0.5': 0, 'c74|blend_unsafe|sep0.12|margin0.005|alpha0.75': 0, 'c75|repel_unsafe|sep0.12|margin0.005|strength0.25': 0, 'c76|repel_unsafe|sep0.12|margin0.005|strength0.5': 0, 'c77|fallback_unsafe|sep0.12|margin0.01': 0, 'c78|predicted_safe_only|sep0.12|margin0.01|safe_min_sep0.12': 0, 'c79|blend_unsafe|sep0.12|margin0.01|alpha0.25': 0, 'c80|blend_unsafe|sep0.12|margin0.01|alpha0.5': 0, 'c81|blend_unsafe|sep0.12|margin0.01|alpha0.75': 0, 'c82|repel_unsafe|sep0.12|margin0.01|strength0.25': 0, 'c83|repel_unsafe|sep0.12|margin0.01|strength0.5': 0}`
- mixed_group_selection: `{'mixed_group_count': 0, 'group_count': 5244, 'mixed_group_rate': 0.0, 'mixed_row_count': 0, 'mixed_row_rate': 0.0}`

## Test Once Metrics

| all | t50 | t100 raw | hard/failure | easy | switch | near base/final |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 24.70% | 22.36% | 14.35% | 23.88% | -25.64% | 58.81% | 1.94%/1.44% |

## Mode Comparison

| mode | val score | test score | all | t50 | t100 raw | hard | easy | mixed groups | buckets val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `global` | 1.890212 | 0.979165 | 24.72% | 22.36% | 14.35% | 23.89% | -25.63% | 0.00% | `{'0': 23788}` |
| `domain_horizon` | 1.893497 | 0.978820 | 24.70% | 22.36% | 14.35% | 23.88% | -25.64% | 0.00% | `{'0': 23788}` |
| `domain_horizon_risk3` | 1.865972 | 0.964178 | 24.28% | 22.14% | 14.35% | 23.39% | -25.52% | 0.00% | `{'0': 907, '1': 11841, '2': 11040}` |
| `domain_horizon_risk4` | 1.860468 | 0.972061 | 24.42% | 22.47% | 14.35% | 23.56% | -25.61% | 0.00% | `{'0': 907, '1': 7015, '2': 7486, '3': 8380}` |

## Delta vs Prior

| prior | all | t50 | t100 raw | hard | easy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Stage42-DI` | -0.01% | 0.00% | 0.00% | -0.01% | -0.01% |
| `Stage42-EX` | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

## Risk Score Stats

- risk_score_stats: `{'val_group_min': 0.0, 'val_group_median': 0.10248438470694114, 'val_group_max': 2.786046186134923, 'test_group_min': 0.0, 'test_group_median': 0.09259236848859972, 'test_group_max': 3.596187907089248, 'val_unique_group_scores': 3847, 'test_unique_group_scores': 2961}`

## Bootstrap CI

| slice | low | mid | high | n |
| --- | ---: | ---: | ---: | ---: |
| `all` | 0.243857 | 0.247105 | 0.249935 | 47458 |
| `t50` | 0.219062 | 0.223574 | 0.227887 | 11538 |
| `t100_raw_frame_diagnostic` | 0.137738 | 0.143520 | 0.149245 | 7048 |
| `hard_failure` | 0.235352 | 0.238698 | 0.241841 | 35076 |
| `easy_degradation` | -0.358494 | -0.344691 | -0.331088 | 11192 |

## Interpretation

- Stage42-EY tests whether EX failed because binary group-risk saturated.
- Continuous risk buckets are validation-derived and frozen before test.
- If EY does not beat Stage42-DI, risk-adaptive repair is not currently a better path than DI's global group-consistency repair.
- This remains source-level raw-frame 2.5D evidence, not metric/seconds-level, true 3D, Stage5C, or SMC evidence.

## Gate

| gate | pass |
| --- | ---: |
| `repair_candidates_evaluated` | True |
| `continuous_group_risk_built` | True |
| `risk_buckets_non_degenerate` | True |
| `adaptive_modes_evaluated` | True |
| `validation_only_mode_selection` | True |
| `selected_mode_recorded` | True |
| `test_all_positive_vs_floor` | True |
| `test_t50_positive_vs_floor` | True |
| `test_hard_positive_vs_floor` | True |
| `easy_degradation_under_2pct` | True |
| `near005_not_worse_than_base` | True |
| `group_consistent_selection` | True |
| `beats_stage42_di_all` | False |
| `beats_stage42_di_hard` | False |
| `no_leakage_pass` | True |
| `no_metric_seconds_overclaim` | True |
| `stage5c_false` | True |
| `smc_false` | True |
