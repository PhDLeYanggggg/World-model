# Stage43-CX T100 Residual Admissibility Group Stress

- source: `fresh_stage43_cx_t100_residual_admissibility_group_stress`
- result_source: `fresh_t100_multi_source_group_stress`
- verdict: `stage43_cx_t100_group_stress_fragile_keep_diagnostic`
- gate: `10 / 10`
- group stress verdict: `multi_source_group_stress_fragile_keep_diagnostic`
- deploy on current heldout t100: `False`

## Aggregate

- all replay exact: `True`
- all group exclusions positive: `False`
- min without any group t100 mean: `0.000413`
- source group flip count max: `0`
- scene group flip count max: `1`
- domain pair flip count max: `0`

### Source-group removals from first seed

| group | labels | group rows | group t100 | without-group t100 | flips |
| --- | ---: | ---: | ---: | ---: | --- |
| `source_top5_positive_gain` | `5` | `5789` | `0.001166` | `0.000114` | `False` |
| `source_top_half_positive_gain` | `6` | `6237` | `0.001107` | `0.000171` | `False` |
| `source_top3_positive_gain` | `3` | `4064` | `0.001224` | `0.000403` | `False` |
| `source_top2_positive_gain` | `2` | `3101` | `0.001375` | `0.000475` | `False` |
| `source_all_negative_slices` | `3` | `2232` | `-0.000993` | `0.001084` | `False` |

### Scene-group removals from first seed

| group | labels | group rows | group t100 | without-group t100 | flips |
| --- | ---: | ---: | ---: | ---: | --- |
| `scene_top5_positive_gain` | `5` | `7526` | `0.001001` | `-0.000056` | `True` |
| `scene_top3_positive_gain` | `3` | `5865` | `0.001155` | `0.000117` | `False` |
| `scene_top2_positive_gain` | `2` | `5402` | `0.001136` | `0.000469` | `False` |
| `scene_all_negative_slices` | `3` | `2232` | `-0.000993` | `0.001084` | `False` |

### Domain-pair removals from first seed

| group | labels | group rows | group t100 | without-group t100 | flips |
| --- | ---: | ---: | ---: | ---: | --- |
| `domain_remove_ETH_UCY+TrajNet` | `2` | `8300` | `0.000866` | `0.000724` | `False` |
| `domain_remove_TrajNet+UCY` | `2` | `4667` | `0.000836` | `0.000845` | `False` |
| `domain_remove_ETH_UCY+UCY` | `2` | `7033` | `0.000815` | `0.000901` | `False` |

## Interpretation

- This is a stricter grouped source/scene stress audit for the tiny CU/CV/CW t100 residual-admissibility signal.
- The grouped scene stress exposes fragility, so this remains a diagnostic result rather than a t100 deployment change.
- Future endpoints/full waypoints remain labels only; inference inputs are causal.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
