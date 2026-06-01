# Stage43-AF Scene-Proxy Counterfactual Ablation

- source: `fresh_stage43_af_scene_proxy_counterfactual_ablation`
- result_source: `fresh_replay_same_route_counterfactual_model_family_ablation`
- gate: `12 / 12`
- verdict: `stage43_af_scene_proxy_counterfactual_contribution_pass`
- route counts: `{'floor': 7962, 'stage43_m': 752, 'stage43_ab': 7286}`

## What Was Compared

Stage43-AF replays the Stage43-AE validation-selected route and then removes the scene-proxy branch counterfactually. On the same rows and same route, any `Stage43-AB` scene-proxy selection is replaced by either `Stage43-M` or the original floor.

This is a same-route model-family ablation: Stage43-M and Stage43-AB are separately trained models. It is not a full factorial retraining of every module.

## Actual AE vs No-Scene Counterfactual

| metric | AE actual | no scene -> Stage43-M | scene-proxy contribution |
| --- | ---: | ---: | ---: |
| all full-waypoint ADE | `23.95%` | `15.72%` | `8.24%` |
| t50 full-waypoint ADE | `37.16%` | `16.45%` | `20.71%` |
| t50 endpoint FDE | `44.63%` | `28.82%` | `15.81%` |
| hard/failure | `23.38%` | `14.14%` | `9.23%` |
| t100 raw-frame diagnostic | `0.00%` | `0.00%` | `0.00%` |
| easy degradation | `0.00%` | `0.00%` | `0.00%` |

## Horizon Contribution vs No-Scene

| horizon | all contribution | t50 contribution | hard contribution | easy contribution |
| --- | ---: | ---: | ---: | ---: |
| `10` | `0.00%` | `0.00%` | `0.00%` | `0.00%` |
| `25` | `12.01%` | `0.00%` | `15.26%` | `0.00%` |
| `50` | `20.71%` | `20.71%` | `20.71%` | `0.00%` |
| `100` | `0.00%` | `0.00%` | `0.00%` | `0.00%` |

## Domain Contribution vs No-Scene

| domain | all contribution | t50 contribution | hard contribution | easy contribution |
| --- | ---: | ---: | ---: | ---: |
| `ETH_UCY` | `8.33%` | `22.40%` | `9.18%` | `0.00%` |
| `TrajNet` | `8.71%` | `23.08%` | `9.33%` | `0.00%` |
| `UCY` | `7.39%` | `12.11%` | `9.50%` | `0.00%` |

## Interpretation

The scene-proxy branch contributes positive `t+50` and endpoint lift under the same Stage43-AE safety route. The result does not claim a uniform all/hard improvement over every alternative: AE is a slice-safe/t50-focused deployment contract, while AC remains stronger for some all/hard objectives but has caveated easy slices.

## Boundary

- Dataset-local/raw-frame 2.5D evidence only.
- Future waypoints are labels/eval only, not input.
- This is a same-route counterfactual model-family ablation, not a full retrained factorial ablation.
- No metric/seconds claim, no Stage5C, no SMC.

## Gate

| gate | passed |
| --- | --- |
| stage43_ae_precondition_pass | True |
| fresh_same_route_counterfactual_replay | True |
| scene_proxy_route_present | True |
| counterfactual_no_scene_built | True |
| scene_proxy_t50_lift_positive | True |
| scene_proxy_endpoint_t50_lift_positive | True |
| scene_proxy_hard_or_all_lift_positive | True |
| actual_easy_preserved | True |
| actual_t100_floor_guarded | True |
| tradeoff_reported_not_overclaimed | True |
| no_future_or_test_leakage | True |
| no_metric_seconds_stage5c_smc_claim | True |
