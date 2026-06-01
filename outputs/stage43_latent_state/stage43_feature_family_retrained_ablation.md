# Stage43-AH Feature-Family Retrained Ablation

- source: `fresh_stage43_ah_feature_family_retrained_ablation`
- result_source: `fresh_retrained_feature_family_ablation`
- mode: `small`
- gate: `12 / 12`
- verdict: `stage43_ah_feature_family_retrained_ablation_pass`
- positive t50 contribution variants: `['no_baseline_floor']`
- positive hard/all contribution variants: `['no_goal', 'no_baseline_floor']`

## Variants

| variant | features | all | t50 | hard | easy | full-minus-variant all | full-minus-variant t50 | full-minus-variant hard | t50 CI mean | latent var |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_features` | `162` | `36.01%` | `30.31%` | `36.98%` | `7.84%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.4222` |
| `no_history` | `54` | `37.49%` | `33.35%` | `40.28%` | `0.00%` | `-1.48%` | `-3.04%` | `-3.31%` | `-3.07%` | `0.3144` |
| `no_goal` | `135` | `27.29%` | `34.22%` | `32.13%` | `29.37%` | `8.72%` | `-3.91%` | `4.85%` | `-3.93%` | `0.3512` |
| `no_neighbor_interaction` | `157` | `39.58%` | `38.81%` | `40.20%` | `0.00%` | `-3.57%` | `-8.50%` | `-3.22%` | `-8.54%` | `0.2834` |
| `no_baseline_floor` | `144` | `29.11%` | `11.40%` | `32.31%` | `35.13%` | `6.91%` | `18.91%` | `4.67%` | `18.89%` | `0.3472` |
| `no_domain` | `159` | `39.79%` | `40.68%` | `41.71%` | `8.13%` | `-3.78%` | `-10.37%` | `-4.74%` | `-10.34%` | `0.3018` |

## Interpretation

Stage43-AH fresh-trains feature-family removal variants under the same protected full-waypoint latent dynamics protocol. A positive full-minus-variant value means the full feature family helped relative to a retrained model without that family.

This is contribution evidence, not a deployment policy: some positive contribution variants still have high easy harm, and history/neighbor/domain removal can outperform full_features in this single-seed small run.

This is stronger than inference masking, but still not a complete all-module factorial ablation and not multi-seed medium evidence.

## Boundary

- Dataset-local/raw-frame 2.5D only.
- Future waypoints are labels/eval only.
- No metric/seconds claim, no Stage5C, no SMC.

## Gate

| gate | passed |
| --- | --- |
| fresh_retrained_ablation | True |
| full_features_baseline_retrained | True |
| at_least_four_feature_family_ablations | True |
| not_inference_masking | True |
| history_or_baseline_family_contribution_found | True |
| at_least_two_feature_families_show_contribution | True |
| bootstrap_or_resampling_recorded | True |
| latent_noncollapse | True |
| checkpoints_not_committed | True |
| no_future_or_test_leakage | True |
| no_metric_seconds_stage5c_smc_claim | True |
| not_overclaimed_full_factorial | True |
