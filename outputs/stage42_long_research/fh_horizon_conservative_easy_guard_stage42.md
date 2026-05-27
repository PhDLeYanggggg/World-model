# Stage42-FN FH Horizon Conservative Easy Guard

- source: `fresh_stage42_fh_horizon_conservative_easy_guard`
- generated_at_utc: `2026-05-27T08:44:28.320565+00:00`
- gate: `15 / 15`
- verdict: `stage42_fn_conservative_easy_guard_pass_with_horizon_limit`
- decision: `conservative_guard_partial_keep_stage42_fh_fi_with_horizon_limit`

## Global Test Metrics vs Floor

- all improvement: `34.86%`
- t50 improvement: `29.03%`
- t100 raw-frame diagnostic improvement: `20.19%`
- hard/failure improvement: `32.96%`
- easy degradation: `-37.14%`
- switch rate: `74.28%`
- final near@0.05: `1.24%`

## Conservative Guard Summary

- weak_domain_horizons_before: `['TrajNet|100', 'UCY|100']`
- weak_domain_horizons_after: `['TrajNet|100', 'UCY|100']`
- repaired_horizon_count: `0`
- uniform_horizon_claim_allowed: `False`

| key | mode | replacement | feature | direction | threshold | rows | guard rows |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| `TrajNet|100` | `feature_guard` | `floor` | `path_length` | `le` | 0.375000 | 5608 | 2593 |
| `UCY|100` | `feature_guard` | `fa` | `min_distance` | `le` | 0.125833 | 1440 | 288 |

## Domain-Horizon Robustness After Conservative Guard

| name | rows | all | t50 | t100 raw | hard | easy | near delta vs FC CI high | robust | weak reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TrajNet|10` | 12342 | 64.69% | 0.00% | 0.00% | 66.10% | -50.55% | -0.05% | True | `none` |
| `TrajNet|25` | 10770 | 37.10% | 0.00% | 0.00% | 30.04% | -49.63% | -0.32% | True | `none` |
| `TrajNet|50` | 9198 | 29.35% | 29.35% | 0.00% | 29.35% | -21.86% | -1.50% | True | `none` |
| `TrajNet|100` | 5608 | 17.70% | 0.00% | 17.70% | 17.70% | 2.75% | -0.36% | False | `easy_ci_exceeds_2pct` |
| `UCY|10` | 3060 | 75.04% | 0.00% | 0.00% | 74.74% | -61.35% | -0.78% | True | `none` |
| `UCY|25` | 2700 | 22.60% | 0.00% | 0.00% | 2.39% | -60.18% | 0.00% | True | `none` |
| `UCY|50` | 2340 | 27.90% | 27.90% | 0.00% | 27.90% | -4.72% | -0.60% | True | `none` |
| `UCY|100` | 1440 | 27.82% | 0.00% | 27.82% | 27.82% | -2.10% | -1.25% | False | `easy_ci_exceeds_2pct` |

## Interpretation

- Stage42-FN is a validation-only conservative easy guard after FM repaired only one weak horizon.
- If weak horizon slices remain, uniform horizon robustness remains blocked.
- No test threshold tuning, no future endpoint input, no central velocity, no Stage5C, no SMC, no metric/seconds-level claim.
