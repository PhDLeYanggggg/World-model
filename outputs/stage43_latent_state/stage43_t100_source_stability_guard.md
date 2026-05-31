# Stage43-R T100 Source-Stability Guard

- source: `fresh_stage43_r_t100_source_stability_guard`
- result_source: `fresh_validation_source_stable_t100_guard`
- gate: `13 / 13`
- verdict: `stage43_r_source_stable_h100_guard_blocks_t100_false_positive`
- full-test rows: `89736`
- h100 status: `h100_blocked_insufficient_source_stability`
- h100 allowed rules: `none`
- blocks Stage43-Q false positive: `True`

## Selected Validation Trial

- target: `residual`
- train filter: `t50t100`
- l2: `10000.0`
- min source count: `2`
- min source rows: `100`

## Deployment Metrics

- full-waypoint ADE improvement: `50.25%`
- t50 full-waypoint ADE improvement: `51.23%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure ADE improvement: `47.88%`
- easy degradation: `0.00%`
- switch rate: `70.45%`

## Bootstrap CI

- bootstrap n: `1000`
- all ADE CI: `[49.96%, 50.56%]`
- t50 ADE CI: `[50.76%, 51.70%]`
- t100 ADE CI: `[0.00%, 0.00%]`

## Validation H100 Source-Stability Table

| family|horizon | rows | agg ADE lift | agg easy | source count | safe source count | reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ETH_UCY|100 | 2560 | -1588.24% | 9134.44% | 1 | 0 | `blocked_insufficient_validation_source_count` |
| TrajNet_crowds|100 | 5608 | 2.50% | 3.75% | 1 | 0 | `blocked_insufficient_validation_source_count` |
| UCY|100 | 7128 | 1.32% | 0.61% | 1 | 1 | `blocked_insufficient_validation_source_count` |

## Interpretation

Stage43-R adds a validation-only source-stability guard for h100. Stage43-Q allowed UCY|100 from a single validation source, but that test candidate produced negative t100 and high easy harm. The source-stability guard blocks such singleton-source h100 deployment, so t100 remains fallback-only and the blocker is now localized to insufficient source-stable h100 evidence.

Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.
