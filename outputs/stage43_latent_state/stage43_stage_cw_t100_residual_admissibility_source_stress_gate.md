# Stage43-CW Gate

- verdict: `stage43_cw_t100_source_stress_survives_single_exclusion_diagnostic`
- passed: `10 / 10`
- stress verdict: `source_scene_stress_survives_single_exclusion`
- deploy on current heldout t100: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass |
| --- | --- |
| `stage43_cv_precondition_present` | `True` |
| `fresh_source_scene_stress` | `True` |
| `three_seed_replay` | `True` |
| `replay_diff_zero` | `True` |
| `source_scene_domain_exclusion_tables_present` | `True` |
| `single_source_exclusion_positive_or_fragility_reported` | `True` |
| `diagnostic_not_deployed` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
