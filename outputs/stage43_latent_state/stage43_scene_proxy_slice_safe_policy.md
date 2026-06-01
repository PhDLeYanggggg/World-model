# Stage43-AE Scene-Proxy Slice-Safe Policy

- source: `fresh_stage43_ae_scene_proxy_slice_safe_policy`
- result_source: `fresh_validation_selected_slice_safe_three_route_policy`
- gate: `14 / 14`
- verdict: `stage43_ae_slice_safe_scene_proxy_candidate`
- deploy slice-safe scene proxy: `True`

## Selected Policy

- policy: `{'family': 'ab_h25_h50_floor_else', 'gain_threshold': 0.0, 'harm_threshold': 1.0, 'failure_threshold': 0.1, 'ab_reject_fallback': 'stage43_m', 'selected_on': 'validation_only', 'test_threshold_tuning': False, 'uses_easy_label_at_inference': False, 'stage43_ad_structural_caveat_guard': True}`
- validation objective: `1.216355`
- route counts: `{'floor': 7962, 'stage43_m': 752, 'stage43_ab': 7286}`

## Test Metrics

- full-waypoint ADE vs floor: `23.95%`; delta vs Stage43-M: `-5.81%`; delta vs AC: `-17.22%`
- t50 ADE vs floor: `37.16%`; delta vs Stage43-M: `20.71%`; delta vs AC: `1.74%`
- hard/failure vs floor: `23.38%`; delta vs Stage43-M: `-5.38%`; delta vs AC: `-18.96%`
- t100 raw-frame diagnostic: `0.00%`
- easy degradation overall/domain/horizon max: `0.00%` / `0.00%` / `0.00%`
- min domain all improvement: `15.34%`

## Domain Diagnostics

| domain | all | t50 | hard | easy | floor | Stage43-M | Stage43-AB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | `22.91%` | `33.61%` | `22.22%` | `0.00%` | `50.04%` | `3.97%` | `45.99%` |
| `TrajNet` | `15.34%` | `42.93%` | `19.06%` | `0.00%` | `51.49%` | `7.87%` | `40.64%` |
| `UCY` | `35.75%` | `50.49%` | `34.76%` | `0.00%` | `45.94%` | `7.05%` | `47.01%` |

## Horizon Diagnostics

| horizon | all | t50 | t100 | hard | easy | floor | Stage43-M | Stage43-AB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `100.00%` | `0.00%` | `0.00%` |
| `25` | `52.59%` | `0.00%` | `0.00%` | `56.79%` | `0.00%` | `0.00%` | `5.38%` | `94.62%` |
| `50` | `37.16%` | `37.16%` | `0.00%` | `37.16%` | `0.00%` | `0.00%` | `13.67%` | `86.33%` |
| `100` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `100.00%` | `0.00%` | `0.00%` |

## Interpretation

Stage43-AE adds a validation-selected three-route safety policy: original floor, Stage43-M protected latent policy, and Stage43-AB scene-proxy latent policy. This directly addresses the Stage43-AD TrajNet/h10/h100 easy-safety caveats without using the easy label as an inference input.

## Boundary

- This is still dataset-local/raw-frame 2.5D evidence.
- t100 is guarded to the floor and remains diagnostic, not solved.
- No future endpoint/waypoint input, no central velocity, no test endpoint goals, no test threshold tuning.
- No metric/seconds claim, no Stage5C, no SMC.

## Gate

| gate | passed |
| --- | --- |
| stage43_ac_available | True |
| stage43_ad_caveat_audit_available | True |
| fresh_validation_selected_policy | True |
| three_route_policy_used | True |
| stage43_ad_structural_caveat_guard | True |
| overall_easy_preserved | True |
| domain_easy_preserved | True |
| horizon_easy_preserved | True |
| all_powered_domains_positive | True |
| core_lift_vs_stage43_m | True |
| t50_still_positive | True |
| t100_guarded_to_nonnegative_floor | True |
| no_future_or_test_leakage | True |
| no_metric_seconds_stage5c_smc_claim | True |
