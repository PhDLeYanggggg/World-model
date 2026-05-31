# Stage43-P Tail-Horizon Full-Waypoint Adapter

- source: `fresh_stage43_p_tail_horizon_waypoint_adapter`
- result_source: `fresh_train_val_selected_tail_horizon_adapter`
- gate: `13 / 13`
- verdict: `stage43_p_tail_horizon_adapter_pass_t100_still_fallback`
- full-test rows: `89736`

## Selected Model

- target: `direct`
- train filter: `t50t100`
- l2: `1000.0`
- train rows: `58845`
- allowed rules: `TrajNet_crowds|10, TrajNet_crowds|25, TrajNet_crowds|50, UCY|10, UCY|25, UCY|50`
- h100 allowed by validation contract: `False`

## Full-Test Metrics

- full-waypoint ADE improvement: `50.25%`
- endpoint FDE improvement: `51.15%`
- t50 full-waypoint ADE improvement: `51.23%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure full-waypoint ADE improvement: `47.88%`
- easy degradation: `0.00%`
- switch rate: `70.45%`

## Delta vs Stage43-O

- all ADE improvement delta: `18.92%`
- t50 delta: `36.10%`
- t100 delta: `0.00%`
- hard/failure delta: `18.70%`
- easy degradation delta: `0.00%`

## Bootstrap CI

- bootstrap n: `1000`
- all ADE CI: `[49.96%, 50.53%]`
- t50 ADE CI: `[50.76%, 51.74%]`
- hard/failure ADE CI: `[47.52%, 48.22%]`
- easy degradation CI: `[0.00%, 0.00%]`

## Horizon Breakdown

| horizon | rows | ADE lift | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: |
| 10 | 26132 | 61.41% | 0.00% | 85.24% |
| 100 | 18070 | 0.00% | 0.00% | 0.00% |
| 25 | 23780 | 63.21% | 0.00% | 88.60% |
| 50 | 21754 | 51.23% | 0.00% | 91.33% |

## Interpretation

Stage43-P trains a real train-split full-waypoint adapter for the tail horizons and selects deployment rules only on validation. It materially improves all/t50/hard over Stage43-O while preserving easy cases. The h100 contract blocks every h100 switch because validation h100 support is not uniformly safe; t100 is therefore not solved, only kept non-harmful by fallback.

Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.
