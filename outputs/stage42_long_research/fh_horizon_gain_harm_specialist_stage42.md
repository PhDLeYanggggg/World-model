# Stage42-FO FH Horizon Gain/Harm Specialist

- source: `fresh_stage42_fh_horizon_gain_harm_specialist`
- generated_at_utc: `2026-05-27T08:55:50.852767+00:00`
- gate: `16 / 16`
- verdict: `stage42_fo_gain_harm_specialist_pass_with_horizon_limit`
- decision: `gain_harm_specialist_partial_keep_stage42_fh_fi_with_horizon_limit`

## Global Test Metrics vs Floor

- all improvement: `35.20%`
- t50 improvement: `29.03%`
- t100 raw-frame diagnostic improvement: `21.14%`
- hard/failure improvement: `33.35%`
- easy degradation: `-37.10%`
- switch rate: `74.68%`
- final near@0.05: `1.25%`

## Gain/Harm Specialist Summary

- weak_domain_horizons_before: `['TrajNet|100', 'UCY|100']`
- weak_domain_horizons_after: `['TrajNet|100', 'UCY|100']`
- repaired_horizon_count: `0`
- uniform_horizon_claim_allowed: `False`

| key | mode | rows | switch rows |
| --- | --- | ---: | ---: |
| `TrajNet|100` | `gain_harm_model` | 5608 | 1962 |
| `UCY|100` | `keep_fm` | 1440 | 0 |

## Domain-Horizon Robustness After Gain/Harm Specialist

| name | rows | all | t50 | t100 raw | hard | easy | near delta vs FC CI high | robust | weak reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `TrajNet|10` | 12342 | 64.69% | 0.00% | 0.00% | 66.10% | -50.55% | -0.05% | True | `none` |
| `TrajNet|25` | 10770 | 37.10% | 0.00% | 0.00% | 30.04% | -49.63% | -0.32% | True | `none` |
| `TrajNet|50` | 9198 | 29.35% | 29.35% | 0.00% | 29.35% | -21.86% | -1.49% | True | `none` |
| `TrajNet|100` | 5608 | 18.98% | 0.00% | 18.98% | 18.98% | 3.31% | -0.27% | False | `easy_ci_exceeds_2pct` |
| `UCY|10` | 3060 | 75.04% | 0.00% | 0.00% | 74.74% | -61.35% | -0.78% | True | `none` |
| `UCY|25` | 2700 | 22.60% | 0.00% | 0.00% | 2.39% | -60.18% | 0.00% | True | `none` |
| `UCY|50` | 2340 | 27.90% | 27.90% | 0.00% | 27.90% | -4.72% | -0.60% | True | `none` |
| `UCY|100` | 1440 | 27.76% | 0.00% | 27.76% | 27.76% | -1.91% | -1.46% | False | `easy_ci_exceeds_2pct` |

## Interpretation

- Stage42-FO trains a validation-only row-level gain/harm specialist on past/prototype/rollout features.
- If weak horizon slices remain, uniform horizon robustness remains blocked.
- No test threshold tuning, no future endpoint input, no central velocity, no Stage5C, no SMC, no metric/seconds-level claim.
