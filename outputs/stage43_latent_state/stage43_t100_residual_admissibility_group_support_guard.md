# Stage43-CY T100 Residual Admissibility Group Support Guard

- source: `fresh_stage43_cy_t100_residual_admissibility_group_support_guard`
- result_source: `fresh_validation_group_support_guard`
- verdict: `stage43_cy_t100_group_support_guard_no_repair_keep_diagnostic`
- gate: `12 / 12`
- deploy on current heldout t100: `False`

## Aggregate

- all replay exact: `True`
- selected variants: `['source_val_positive', 'source_val_positive', 'source_val_positive']`
- base t100 mean: `0.001174`
- guarded t100 mean: `0.001156`
- base min without group t100 mean: `0.000413`
- guarded min without group t100 mean: `0.000355`
- delta min without group t100 mean: `-0.000058`
- group fragility reduced: `False`
- guarded easy degradation max: `0.000000`
- guarded switch rate mean: `0.064633`

## First Seed

- seed: `4323`
- eligible label counts: `{'source': 12, 'scene': 10, 'domain': 3}`
- selected variant: `source_val_positive`
- validation objective: `0.001036`
- test delta vs base: `{'t100': 0.0, 'hard_failure': 0.0, 'easy_degradation': 0.0, 'switch_rate': 0.0, 'min_without_group_t100': 0.0}`

## Interpretation

- This is a validation-selected group-support guard after the Stage43-CX grouped scene/source stress failure.
- It asks whether restricting t100 residual switches to source/scene/domain groups that were positive on validation can reduce grouped fragility on test.
- In this run, group fragility reduced: `False`.
- The guard is diagnostic and does not change current heldout t100 deployment.
- Future endpoints/full waypoints remain labels only; inference inputs and guard metadata are causal or split metadata.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
