# Stage43-DD T100 Policy-Distilled Group-Stability Guard

- source: `fresh_stage43_dd_t100_policy_distilled_group_stability_guard`
- result_source: `fresh_validation_group_support_guard_on_policy_distilled_head`
- verdict: `stage43_dd_t100_policy_distilled_group_guard_mean_improves_dc_seed_fragile`
- gate: `12 / 12`
- deploy on current heldout t100: `False`

## Aggregate

- base DC t100 mean: `0.002167`
- guarded t100 mean: `0.002176`
- base min-without-group mean: `0.000058`
- guarded min-without-group mean: `0.000093`
- guarded easy degradation max: `0.000000`
- selected variants: `['scene_val_positive', 'scene_val_positive', 'source_val_positive']`
- group fragility reduced: `True`
- all guarded min-without-group positive: `False`
- beats DC t100 mean: `True`
- beats CZ t100 mean: `True`
- beats DC min-without-group mean: `True`
- beats CZ min-without-group mean: `False`

## Per Seed

| seed | variant | base t100 | guarded t100 | base min-without | guarded min-without | easy | switch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4323` | `scene_val_positive` | `0.002556` | `0.002556` | `-0.000491` | `-0.000491` | `0.000000` | `0.121200` |
| `4331` | `scene_val_positive` | `0.002277` | `0.002306` | `0.000144` | `0.000247` | `0.000000` | `0.090200` |
| `4337` | `source_val_positive` | `0.001666` | `0.001666` | `0.000521` | `0.000521` | `0.000000` | `0.094200` |

## Interpretation

- This is a validation-only group-support guard over the policy-distilled DC head.
- It tests whether DC's higher t100 mean can be made group-stable without test threshold tuning.
- In this run, the mean guard effect is positive but at least one seed still has negative worst-group t100, so this is not a deployment repair.
- Future waypoints remain labels/eval only; inference inputs remain causal.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
