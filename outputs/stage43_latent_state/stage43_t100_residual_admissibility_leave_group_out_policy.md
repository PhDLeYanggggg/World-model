# Stage43-CZ T100 Leave-Group-Out Robust Admissibility Policy

- source: `fresh_stage43_cz_t100_residual_admissibility_leave_group_out_policy`
- result_source: `fresh_leave_group_out_robust_policy_search`
- verdict: `stage43_cz_t100_leave_group_out_policy_reduces_fragility_diagnostic`
- gate: `13 / 13`
- deploy on current heldout t100: `False`

## Aggregate

- all replay exact: `True`
- selected modes: `['leave_group_out_robust', 'leave_group_out_robust', 'leave_group_out_robust']`
- safe candidate count min: `849`
- original t100 mean: `0.001174`
- robust t100 mean: `0.001841`
- original min without group t100 mean: `0.000413`
- robust min without group t100 mean: `0.000995`
- delta min without group t100 mean: `0.000582`
- delta scene group flip count mean: `-0.333333`
- group fragility reduced: `True`
- robust easy degradation max: `0.000000`
- robust switch rate mean: `0.111133`

## First Seed

- seed: `4323`
- selected policy: `{'alpha': 0.75, 'alpha_index': 5, 'gain_threshold': 0.8, 'harm_threshold': 0.35, 'delta_threshold': 0.0, 'force_easy_floor': False, 'selection_mode': 'leave_group_out_robust'}`
- validation objective: `0.003018`
- test delta vs original: `{'t100': 0.0008044714472403847, 'hard_failure': 0.0008044714472403847, 'easy_degradation': 0.0, 'switch_rate': 0.030600000000000002, 'min_without_group_t100': 0.0008632957205741976, 'scene_group_flip_count': -1.0}`

## Interpretation

- This step changes the validation selection objective itself: policies are rewarded for positive t100 while also surviving leave-group-out source/scene/domain stress.
- It is stricter than the Stage43-CY group whitelist, but it still does not retrain the admissibility head.
- In this run, group fragility reduced: `True`.
- Because this is still policy selection over an existing head, the next repair should train the leave-group-out criterion into the admissibility head and confirm it with stronger heldout/bootstrap evidence.
- Future endpoints/full waypoints remain labels only; inference inputs are causal.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
