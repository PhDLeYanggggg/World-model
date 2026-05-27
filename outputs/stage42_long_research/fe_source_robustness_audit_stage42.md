# Stage42-FG FE Source / Domain / Horizon Robustness Audit

- source: `fresh_stage42_fe_source_robustness_audit`
- generated_at_utc: `2026-05-27T07:21:39.854868+00:00`
- gate: `11 / 12`
- verdict: `stage42_fg_fe_source_robustness_partial`
- FE policy hash: `a78db26aa155b38799f5b866f32a2d205018adf2054d9409a016da3163328dff`

## Summary

- test_rows: `47458`
- domain_count: `2`
- source_count: `3`
- domain_horizon_count: `8`
- robust_domains: `['TrajNet']`
- weak_domains: `['UCY']`
- weak_domain_horizons: `['TrajNet|100', 'UCY|10', 'UCY|25', 'UCY|50', 'UCY|100']`
- weak_sources: `['TrajNet/Train/crowds/crowds_zara03.txt']`
- broad_uniform_source_claim_allowed: `False`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-FF 已冻结 Stage42-FE policy 并做 exact replay + 2000-bootstrap。
- Stage42-FG 不重新训练、不重新选择 policy、不调 test threshold；它审计 FE 在 domain/source/horizon/scene 切片上的稳健性。
- 弱 source / weak slice 必须显式报告；不能把 global positive 包装成每个外部源都 positive。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Domain Robustness

| name | rows | all | t50 | t100 raw | hard | easy | near delta vs FC CI high | robust | weak reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TrajNet` | 37918 | 34.45% | 29.63% | 18.59% | 32.64% | -36.68% | -0.59% | True | `none` |
| `UCY` | 9540 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% | False | `all_ci_not_positive, hard_ci_not_positive` |

## Domain-Horizon Robustness

| name | rows | all | t50 | t100 raw | hard | easy | near delta vs FC CI high | robust | weak reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TrajNet|10` | 12342 | 64.27% | 0.00% | 0.00% | 65.81% | -49.70% | -0.05% | True | `none` |
| `TrajNet|25` | 10770 | 38.33% | 0.00% | 0.00% | 32.90% | -50.03% | -0.46% | True | `none` |
| `TrajNet|50` | 9198 | 29.63% | 29.63% | 0.00% | 29.63% | -21.53% | -1.41% | True | `none` |
| `TrajNet|100` | 5608 | 18.59% | 0.00% | 18.59% | 18.59% | 2.99% | -0.21% | False | `easy_ci_exceeds_2pct` |
| `UCY|10` | 3060 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% | False | `all_ci_not_positive, hard_ci_not_positive` |
| `UCY|25` | 2700 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% | False | `all_ci_not_positive, hard_ci_not_positive` |
| `UCY|50` | 2340 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% | False | `all_ci_not_positive, hard_ci_not_positive` |
| `UCY|100` | 1440 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% | False | `all_ci_not_positive, hard_ci_not_positive` |

## Source Robustness

| name | rows | all | t50 | t100 raw | hard | easy | near delta vs FC CI high | robust | weak reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TrajNet/Test/crowds/students002.txt` | 765 | 67.51% | 63.42% | 0.00% | 67.51% | -61.05% | 0.00% | True | `none` |
| `TrajNet/Train/crowds/crowds_zara03.txt` | 9540 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | 0.00% | False | `all_ci_not_positive, hard_ci_not_positive` |
| `TrajNet/Train/crowds/students003.txt` | 37153 | 33.44% | 28.80% | 18.59% | 31.40% | -35.45% | -0.61% | True | `none` |

## Weak Scene Rows

| scene | rows | all | t50 | t100 raw | hard | easy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `UCY_crowds` | 9540 | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% |
| `TrajNet_crowds` | 37918 | 34.45% | 29.63% | 18.59% | 32.64% | -36.68% |

## Interpretation

- Stage42-FG is an audit, not a new policy selection or new training run.
- Global FE/FF evidence is strong, but source/domain/horizon weak slices remain visible and must be cited as limitations.
- Broad uniform source-level claims are allowed only if powered source slices have no weak failures.
- No Stage5C, SMC, metric/seconds-level, true-3D, foundation, or floor-free neural claim is made.
