# Stage42-FK FH Horizon Weak-Slice Validation Repair

- source: `fresh_stage42_fh_horizon_weak_slice_repair`
- generated_at_utc: `2026-05-27T08:04:55.884902+00:00`
- gate: `15 / 15`
- verdict: `stage42_fk_fh_horizon_weak_slice_repair_pass_with_horizon_limit`
- decision: `horizon_repair_partial_keep_stage42_fh_fi_with_horizon_limit`

## Global Test Metrics vs Floor

- all improvement: `35.18%`
- t50 improvement: `28.97%`
- t100 raw-frame diagnostic improvement: `21.13%`
- hard/failure improvement: `33.33%`
- easy degradation: `-36.88%`
- switch rate: `74.66%`
- final near@0.05: `1.25%`

## Horizon Repair Summary

- weak_domain_horizons_before: `['TrajNet|100', 'UCY|50', 'UCY|100']`
- weak_domain_horizons_after: `['TrajNet|100', 'UCY|50', 'UCY|100']`
- repaired_horizon_count: `0`
- uniform_horizon_claim_allowed: `False`
- applied_overrides: `{'TrajNet|100': {'candidate': 'fb', 'rows': 5608, 'reason': 'validation_safe_best_score'}, 'UCY|50': {'candidate': 'fh', 'rows': 2340, 'reason': 'validation_safe_best_score'}, 'UCY|100': {'candidate': 'fa', 'rows': 1440, 'reason': 'validation_safe_best_score'}}`

## Validation Choices

| key | candidate | reason | score |
| --- | --- | --- | ---: |
| `TrajNet|100` | `fb` | `validation_safe_best_score` | 0.273641 |
| `UCY|50` | `fh` | `validation_safe_best_score` | 0.345051 |
| `UCY|100` | `fa` | `validation_safe_best_score` | 0.345317 |

## Domain-Horizon Robustness After Repair

| name | rows | all | t50 | t100 raw | hard | easy | near delta vs FC CI high | robust | weak reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TrajNet|10` | 12342 | 64.69% | 0.00% | 0.00% | 66.10% | -50.55% | -0.05% | True | `none` |
| `TrajNet|25` | 10770 | 37.10% | 0.00% | 0.00% | 30.04% | -49.63% | -0.32% | True | `none` |
| `TrajNet|50` | 9198 | 29.35% | 29.35% | 0.00% | 29.35% | -21.86% | -1.51% | True | `none` |
| `TrajNet|100` | 5608 | 19.01% | 0.00% | 19.01% | 19.01% | 3.34% | -0.27% | False | `easy_ci_exceeds_2pct` |
| `UCY|10` | 3060 | 75.04% | 0.00% | 0.00% | 74.74% | -61.35% | -0.78% | True | `none` |
| `UCY|25` | 2700 | 22.60% | 0.00% | 0.00% | 2.39% | -60.18% | 0.00% | True | `none` |
| `UCY|50` | 2340 | 27.63% | 27.63% | 0.00% | 27.63% | 0.04% | -0.77% | False | `easy_ci_exceeds_2pct` |
| `UCY|100` | 1440 | 27.63% | 0.00% | 27.63% | 27.63% | -2.40% | -1.25% | False | `easy_ci_exceeds_2pct` |

## Interpretation

- Stage42-FK is a validation-only repair attempt, not a new training run.
- If weak horizon slices remain, uniform horizon robustness remains blocked.
- No Stage5C, SMC, metric/seconds-level, true-3D, foundation, or floor-free neural claim is made.
