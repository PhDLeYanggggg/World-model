# Stage43-AI Feature-Family Multi-Seed Confirmation

- source: `fresh_stage43_ai_feature_family_multiseed_confirmation`
- result_source: `fresh_multiseed_retrained_feature_family_confirmation`
- mode: `small`
- seeds: `[431, 443, 457]`
- gate: `8 / 8`
- verdict: `stage43_ai_feature_family_multiseed_confirmation_pass`
- stable positive t50 variants: `['no_neighbor_interaction', 'no_baseline_floor', 'no_domain']`
- stable positive hard/all variants: `['no_goal', 'no_baseline_floor', 'no_domain']`

## Summary

| variant | mean all | mean t50 | mean hard | mean easy | delta all mean | delta t50 mean | delta hard mean | t50 positive seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_features` | `33.99%` | `27.72%` | `35.01%` | `4.26%` | `0.00%` | `0.00%` | `0.00%` | `0` |
| `no_history` | `38.12%` | `34.61%` | `39.19%` | `0.00%` | `-4.13%` | `-6.88%` | `-4.19%` | `0` |
| `no_goal` | `29.07%` | `27.83%` | `33.91%` | `26.69%` | `4.93%` | `-0.10%` | `1.10%` | `2` |
| `no_neighbor_interaction` | `35.80%` | `23.35%` | `37.52%` | `9.49%` | `-1.81%` | `4.37%` | `-2.52%` | `2` |
| `no_baseline_floor` | `32.91%` | `16.60%` | `35.91%` | `25.33%` | `1.09%` | `11.12%` | `-0.90%` | `2` |
| `no_domain` | `34.35%` | `24.97%` | `36.52%` | `21.59%` | `-0.35%` | `2.76%` | `-1.51%` | `2` |

## Interpretation

Stage43-AI repeats the retrained feature-family ablation across multiple seeds. It is meant to test whether Stage43-AH's module contribution evidence is stable rather than a one-seed artifact.

This remains dataset-local/raw-frame 2.5D evidence. It is not a deployment policy, not metric/seconds evidence, and not Stage5C or SMC execution.

## Gate

| gate | passed |
| --- | --- |
| fresh_multiseed_retrained_confirmation | True |
| at_least_three_seeds | True |
| full_features_and_four_ablations | True |
| baseline_floor_t50_contribution_stable | True |
| at_least_two_stable_module_contributions | True |
| no_future_or_test_leakage | True |
| no_metric_seconds_stage5c_smc_claim | True |
| checkpoints_not_committed | True |
