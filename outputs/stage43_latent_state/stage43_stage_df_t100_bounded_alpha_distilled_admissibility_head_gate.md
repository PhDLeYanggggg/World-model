# Stage43-DF Gate

- verdict: `stage43_df_t100_bounded_alpha_distilled_head_incomplete`
- passed: `14 / 15`
- all min-without-group positive: `False`
- beats DE t100 mean: `False`
- deploy on current heldout t100: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass |
| --- | --- |
| `de_precondition_present` | `True` |
| `fresh_bounded_alpha_distillation_training` | `True` |
| `three_or_more_seeds` | `True` |
| `all_checkpoints_written_not_committed` | `True` |
| `teacher_policy_from_de` | `True` |
| `bounded_alpha_protocol` | `True` |
| `validation_only_policy_selection` | `True` |
| `feature_contract_clean` | `True` |
| `test_once_per_seed` | `True` |
| `easy_preserved` | `True` |
| `all_min_without_group_positive` | `False` |
| `diagnostic_not_deployed` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
