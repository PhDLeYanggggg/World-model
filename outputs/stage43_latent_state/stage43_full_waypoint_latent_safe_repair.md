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

- full-waypoint ADE improvement: `39.22%`
- endpoint FDE improvement: `43.87%`
- t50 full-waypoint ADE improvement: `30.06%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure full-waypoint ADE improvement: `39.85%`
- easy degradation: `0.00%`
- switch rate: `64.57%`

## Delta vs Stage43-N

- all ADE improvement delta: `2.48%`
- t50 delta: `-3.24%`
- t100 delta: `27.88%`
- hard/failure delta: `1.40%`
- easy degradation delta: `-0.47%`

## Horizon Breakdown

| horizon | rows | ADE lift | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: |
| 10 | 26132 | 59.02% | 12.09% | 84.05% |
| 100 | 18070 | 0.00% | 0.00% | 0.00% |
| 25 | 23780 | 44.98% | 0.00% | 83.81% |
| 50 | 21754 | 30.06% | 0.00% | 73.76% |

## Source-Family Breakdown

| source family | rows | ADE lift | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: |
| TrajNet_biwi | 7685 | 0.00% | 0.00% | 0.00% |
| TrajNet_crowds | 9540 | 58.52% | 0.00% | 77.04% |
| TrajNet_mot | 1926 | 0.00% | 0.00% | 0.00% |
| UCY | 70585 | 40.42% | 1.18% | 71.67% |

## Interpretation

The repair removes the Stage43-N negative source and long-horizon harm by falling back where validation support is insufficient or validation h100 is unsafe. This is a safer protected policy, not proof that t100 is solved: t100 is repaired to fallback-level `0.0`, not positive transfer.

Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.
