# Stage43-DD Gate

- verdict: `stage43_dd_t100_policy_distilled_group_guard_mean_improves_dc_seed_fragile`
- passed: `12 / 12`
- group fragility reduced: `True`
- all guarded min-without-group positive: `False`
- beats DC t100 mean: `True`
- deploy on current heldout t100: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass |
| --- | --- |
| `stage43_dc_precondition_present` | `True` |
| `fresh_dc_group_stability_guard` | `True` |
| `three_seed_replay` | `True` |
| `replay_diff_zero` | `True` |
| `validation_only_guard_selection` | `True` |
| `guard_variants_evaluated` | `True` |
| `easy_preserved` | `True` |
| `t100_positive_all_seeds` | `True` |
| `diagnostic_not_deployed` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
