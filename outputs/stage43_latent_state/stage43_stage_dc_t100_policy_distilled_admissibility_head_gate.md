# Stage43-DC Gate

- verdict: `stage43_dc_t100_policy_distilled_head_beats_cz_diagnostic`
- passed: `13 / 13`
- beats DA t100 mean: `True`
- beats CZ t100 mean: `True`
- deploy on current heldout t100: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass |
| --- | --- |
| `db_precondition_present` | `True` |
| `fresh_policy_distillation_training` | `True` |
| `three_or_more_seeds` | `True` |
| `all_checkpoints_written_not_committed` | `True` |
| `teacher_policy_from_cz` | `True` |
| `validation_only_policy_selection` | `True` |
| `feature_contract_clean` | `True` |
| `test_once_per_seed` | `True` |
| `easy_preserved` | `True` |
| `diagnostic_not_deployed` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
