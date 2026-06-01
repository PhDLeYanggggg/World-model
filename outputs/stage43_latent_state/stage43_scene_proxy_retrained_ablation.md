# Stage43-AG Scene-Proxy Retrained Ablation

- source: `fresh_stage43_ag_scene_proxy_retrained_ablation`
- result_source: `fresh_retrained_scene_proxy_subset_ablation`
- mode: `small`
- gate: `11 / 11`
- verdict: `stage43_ag_scene_proxy_retrained_ablation_pass`
- best t50 variant: `full_scene`
- best safe t50 variant: `geometry_route`
- best hard variant: `full_scene`
- best all variant: `full_scene`

## Variants

| variant | scene features | all | t50 | hard | easy | delta all vs no-scene | delta t50 vs no-scene | delta hard vs no-scene | latent var |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_scene` | `0` | `36.01%` | `30.30%` | `36.97%` | `7.86%` | `0.00%` | `0.00%` | `0.00%` | `0.4222` |
| `geometry_route` | `9` | `36.44%` | `35.32%` | `37.20%` | `0.00%` | `0.43%` | `5.02%` | `0.22%` | `0.3567` |
| `goal_only` | `9` | `27.87%` | `26.80%` | `28.92%` | `6.78%` | `-8.14%` | `-3.50%` | `-8.06%` | `0.3335` |
| `full_scene` | `14` | `37.06%` | `36.09%` | `37.94%` | `9.59%` | `1.05%` | `5.79%` | `0.97%` | `0.2895` |

## Interpretation

The best t50 scene subset is `full_scene` with delta vs retrained no-scene `5.79%`.
The best safe t50 scene subset is `geometry_route` with delta `5.02%` and easy degradation `0.00%`.

This is a focused retrained scene-proxy subset ablation. It does not replace the broader Stage43-AF same-route deployment counterfactual, and it is not a full all-module factorial ablation. Unsafe higher-lift variants are reported but not treated as deployable evidence.

## Boundary

- Dataset-local/raw-frame 2.5D only.
- Scene proxy is train-only route/goal/context proxy, not raw imagery or verified SDF.
- Future waypoints are labels/eval only.
- No metric/seconds claim, no Stage5C, no SMC.

## Gate

| gate | passed |
| --- | --- |
| fresh_retrained_ablation | True |
| no_scene_baseline_retrained | True |
| multiple_scene_subsets_retrained | True |
| scene_subset_t50_lift_found | True |
| scene_subset_hard_or_all_lift_found | True |
| safe_t50_scene_variant_available | True |
| latent_noncollapse_for_scene_variants | True |
| checkpoints_not_committed | True |
| no_future_or_test_leakage | True |
| no_metric_seconds_stage5c_smc_claim | True |
| not_overclaimed_full_factorial | True |
