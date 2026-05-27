# Stage42-FJ FH Source / Domain / Horizon Robustness Audit

- source: `fresh_stage42_fh_source_robustness_audit`
- generated_at_utc: `2026-05-27T07:54:26.255579+00:00`
- gate: `14 / 14`
- verdict: `stage42_fj_fh_source_robustness_pass`
- FH policy hash: `f1f6e0636167fae8721a3f7195f188dcbe1a83194b04fa0625b378ad38b5aed6`

## Summary

- test_rows: `47458`
- domain_count: `2`
- source_count: `3`
- domain_horizon_count: `8`
- robust_domains: `['TrajNet', 'UCY']`
- weak_domains: `[]`
- robust_domain_horizons: `['TrajNet|10', 'TrajNet|25', 'TrajNet|50', 'UCY|10', 'UCY|25']`
- weak_domain_horizons: `['TrajNet|100', 'UCY|50', 'UCY|100']`
- robust_sources: `['TrajNet/Test/crowds/students002.txt', 'TrajNet/Train/crowds/crowds_zara03.txt', 'TrajNet/Train/crowds/students003.txt']`
- weak_sources: `[]`
- dual_domain_positive_safe_claim_allowed: `True`
- broad_uniform_source_claim_allowed: `True`
- broad_uniform_horizon_claim_allowed: `False`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-FI 已冻结 Stage42-FH policy，并做 exact replay + 2000-bootstrap。
- Stage42-FJ 不重新训练、不重新选择 policy、不调 test threshold；它只审计 frozen FH 在 domain/source/horizon/scene 切片上的稳健性。
- 弱 source / weak slice 必须显式报告；不能把 global positive 包装成每个外部源、每个 horizon 都 positive。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Domain Robustness

| name | rows | all | t50 | t100 raw | hard | easy | near delta vs FC CI high | robust | weak reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TrajNet` | 37918 | 34.21% | 29.35% | 18.47% | 32.36% | -36.75% | -0.57% | True | `none` |
| `UCY` | 9540 | 37.49% | 27.63% | 26.99% | 35.43% | -37.79% | -0.80% | True | `none` |

## Domain-Horizon Robustness

| name | rows | all | t50 | t100 raw | hard | easy | near delta vs FC CI high | robust | weak reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TrajNet|10` | 12342 | 64.69% | 0.00% | 0.00% | 66.10% | -50.55% | -0.05% | True | `none` |
| `TrajNet|25` | 10770 | 37.10% | 0.00% | 0.00% | 30.04% | -49.63% | -0.33% | True | `none` |
| `TrajNet|50` | 9198 | 29.35% | 29.35% | 0.00% | 29.35% | -21.86% | -1.51% | True | `none` |
| `TrajNet|100` | 5608 | 18.47% | 0.00% | 18.47% | 18.47% | 3.08% | -0.18% | False | `easy_ci_exceeds_2pct` |
| `UCY|10` | 3060 | 75.04% | 0.00% | 0.00% | 74.74% | -61.35% | -0.75% | True | `none` |
| `UCY|25` | 2700 | 22.60% | 0.00% | 0.00% | 2.39% | -60.18% | 0.00% | True | `none` |
| `UCY|50` | 2340 | 27.63% | 27.63% | 0.00% | 27.63% | 0.04% | -0.77% | False | `easy_ci_exceeds_2pct` |
| `UCY|100` | 1440 | 26.99% | 0.00% | 26.99% | 26.99% | -3.45% | -1.39% | False | `easy_ci_exceeds_2pct` |

## Source Robustness

| name | rows | all | t50 | t100 raw | hard | easy | near delta vs FC CI high | robust | weak reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TrajNet/Test/crowds/students002.txt` | 765 | 61.26% | 66.76% | 0.00% | 61.26% | -51.24% | 0.00% | True | `none` |
| `TrajNet/Train/crowds/crowds_zara03.txt` | 9540 | 37.49% | 27.63% | 26.99% | 35.43% | -37.79% | -0.80% | True | `none` |
| `TrajNet/Train/crowds/students003.txt` | 37153 | 33.39% | 28.43% | 18.47% | 31.33% | -36.02% | -0.57% | True | `none` |

## Weak Scene Rows

| scene | rows | all | t50 | t100 raw | hard | easy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `TrajNet_crowds` | 37918 | 34.21% | 29.35% | 18.47% | 32.36% | -36.75% |
| `UCY_crowds` | 9540 | 37.49% | 27.63% | 26.99% | 35.43% | -37.79% |

## Interpretation

- Stage42-FJ is a robustness audit, not new training and not policy reselection.
- FH/FI remains frozen; this report decides which claims are allowed at domain/source/horizon granularity.
- Powered weak source or horizon slices block broad uniform source/horizon claims even when global metrics are strong.
- No Stage5C, SMC, metric/seconds-level, true-3D, foundation, or floor-free neural claim is made.
