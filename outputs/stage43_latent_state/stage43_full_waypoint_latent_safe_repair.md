# Stage43-O Full-Waypoint Latent Safe Repair

- source: `fresh_stage43_o_full_waypoint_latent_safe_repair`
- result_source: `fresh_validation_only_safe_repair_from_stage43_m_checkpoint`
- gate: `12 / 12`
- verdict: `stage43_o_safe_repair_pass_t100_fallback_not_positive`
- full-test rows: `89736`

## What changed

Stage43-O does not retrain the latent model and does not tune on test. It uses validation-only source-family/horizon support rules to decide when the Stage43-M latent full-waypoint head is allowed to switch away from the frozen floor.

- allowed rules: `ETH_UCY|25, ETH_UCY|50, TrajNet_crowds|10, TrajNet_crowds|25, TrajNet_crowds|50, UCY|10, UCY|25, UCY|50`
- min validation support rows: `1000`
- max validation easy degradation: `2.00%`

## Full-Test Metrics

- full-waypoint ADE improvement: `31.34%`
- endpoint FDE improvement: `39.25%`
- t50 full-waypoint ADE improvement: `15.13%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure full-waypoint ADE improvement: `29.18%`
- easy degradation: `0.00%`
- switch rate: `55.50%`

## Delta vs Stage43-N

- all ADE improvement delta: `1.92%`
- t50 delta: `-1.47%`
- t100 delta: `16.57%`
- hard/failure delta: `0.52%`
- easy degradation delta: `0.00%`

## Horizon Breakdown

| horizon | rows | ADE lift | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: |
| 10 | 26132 | 50.19% | 0.00% | 73.81% |
| 100 | 18070 | 0.00% | 0.00% | 0.00% |
| 25 | 23780 | 39.48% | 0.00% | 74.08% |
| 50 | 21754 | 15.13% | 0.00% | 59.28% |

## Source-Family Breakdown

| source family | rows | ADE lift | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: |
| TrajNet_biwi | 7685 | 0.00% | 0.00% | 0.00% |
| TrajNet_crowds | 9540 | 48.40% | 0.00% | 78.63% |
| TrajNet_mot | 1926 | 0.00% | 0.00% | 0.00% |
| UCY | 70585 | 32.00% | 0.00% | 59.93% |

## Interpretation

The repair removes the Stage43-N negative source and long-horizon harm by falling back where validation support is insufficient or validation h100 is unsafe. This is a safer protected policy, not proof that t100 is solved: t100 is repaired to fallback-level `0.0`, not positive transfer.

Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.
