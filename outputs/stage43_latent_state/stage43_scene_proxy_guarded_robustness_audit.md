# Stage43-AD Scene-Proxy Guarded Robustness Audit

- source: `fresh_stage43_ad_scene_proxy_guarded_robustness_audit`
- result_source: `fresh_replay_stage43_ac_slice_robustness_audit`
- gate: `12 / 12`
- verdict: `stage43_ad_guarded_scene_proxy_caveated_audit_pass`
- all powered domains positive: `False`

## Overall

- rows: `16000`
- AC full-waypoint ADE vs floor: `41.17%`; delta vs Stage43-M: `11.40%`
- AC t50 ADE vs floor: `35.42%`; delta vs Stage43-M: `18.97%`
- AC hard/failure vs floor: `42.34%`; delta vs Stage43-M: `13.58%`
- AC easy degradation: `0.00%`
- AC t100 raw-frame diagnostic: `-17.79%`; delta vs Stage43-M: `0.00%`

## Domain Table

| domain | rows | AC all | delta all vs M | AC t50 | AC hard | easy | override | caveat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `domain:ETH_UCY` | `12648` | `41.40%` | `11.60%` | `31.93%` | `42.14%` | `0.00%` | `78.36%` | `none` |
| `domain:TrajNet` | `1678` | `19.69%` | `10.52%` | `34.78%` | `35.15%` | `52.99%` | `88.44%` | `easy_degradation_over_2pct` |
| `domain:UCY` | `1674` | `54.41%` | `10.90%` | `51.27%` | `49.69%` | `0.00%` | `85.13%` | `none` |

## Horizon Table

| horizon | rows | AC all | delta all vs M | AC t50 | AC t100 | AC hard | easy | override | caveat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `horizon:10` | `4782` | `61.34%` | `10.73%` | `0.00%` | `0.00%` | `72.51%` | `22.21%` | `100.00%` | `easy_degradation_over_2pct` |
| `horizon:25` | `4182` | `52.67%` | `12.09%` | `0.00%` | `0.00%` | `56.60%` | `0.00%` | `100.00%` | `none` |
| `horizon:50` | `3856` | `35.42%` | `18.97%` | `35.42%` | `0.00%` | `35.42%` | `0.00%` | `100.00%` | `none` |
| `horizon:100` | `3180` | `-17.79%` | `0.00%` | `0.00%` | `-17.79%` | `-17.79%` | `10.40%` | `0.00%` | `non_positive_ac_all, easy_degradation_over_2pct` |

## Caveat Slices

- caveat slice count: `9`
- AC does not claim uniform horizon success. t100 remains raw-frame diagnostic and is guarded by falling back to Stage43-M.

## Gate

| gate | passed |
| --- | --- |
| stage43_ac_candidate_available | True |
| fresh_slice_replay_completed | True |
| domain_table_reported | True |
| horizon_table_reported | True |
| source_table_reported | True |
| overall_easy_preserved | True |
| overall_lift_vs_stage43_m | True |
| t100_caveat_recorded | True |
| powered_domain_status_reported | True |
| weak_slices_not_hidden | True |
| no_future_or_test_leakage | True |
| no_metric_seconds_stage5c_smc_claim | True |
