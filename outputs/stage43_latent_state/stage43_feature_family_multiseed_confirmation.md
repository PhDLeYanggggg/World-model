# Stage43-AI Feature-Family Multi-Seed Confirmation

- source: `fresh_stage43_ai_feature_family_multiseed_confirmation`
- result_source: `fresh_multiseed_retrained_feature_family_confirmation`
- mode: `small`
- seeds: `[431, 443, 457]`
- gate: `8 / 8`
- verdict: `stage43_ai_feature_family_multiseed_confirmation_pass`
- stable positive t50 variants: `['no_baseline_floor']`
- stable positive hard/all variants: `['no_goal', 'no_baseline_floor']`

## Summary

| variant | mean all | mean t50 | mean hard | mean easy | delta all mean | delta t50 mean | delta hard mean | t50 positive seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_features` | `32.48%` | `29.43%` | `33.56%` | `10.31%` | `0.00%` | `0.00%` | `0.00%` | `0` |
| `no_history` | `35.33%` | `33.82%` | `37.92%` | `16.42%` | `-2.84%` | `-4.39%` | `-4.36%` | `1` |
| `no_goal` | `28.14%` | `33.73%` | `32.66%` | `28.43%` | `4.34%` | `-4.30%` | `0.90%` | `1` |
| `no_neighbor_interaction` | `37.11%` | `35.98%` | `38.21%` | `6.29%` | `-4.63%` | `-6.55%` | `-4.65%` | `1` |
| `no_baseline_floor` | `32.91%` | `16.60%` | `35.91%` | `25.33%` | `-0.43%` | `12.83%` | `-2.36%` | `3` |
| `no_domain` | `34.84%` | `34.97%` | `35.98%` | `11.47%` | `-2.36%` | `-5.54%` | `-2.43%` | `1` |

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
