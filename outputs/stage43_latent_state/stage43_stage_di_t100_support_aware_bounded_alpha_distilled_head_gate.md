# Stage43-DI Gate

- verdict: `stage43_di_t100_support_aware_distilled_head_safe_but_no_lift_diagnostic`
- passed: `15 / 15`
- all min-without-group positive: `True`
- beats DH t100 mean: `False`
- deploy on current heldout t100: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass |
| --- | --- |
| `dh_precondition_present` | `True` |
| `fresh_support_aware_head_training` | `True` |
| `three_or_more_seeds` | `True` |
| `all_checkpoints_written_not_committed` | `True` |
| `teacher_policy_from_dh` | `True` |
| `support_aware_validation_selection` | `True` |
| `validation_only_policy_selection` | `True` |
| `feature_contract_clean` | `True` |
| `test_once_per_seed` | `True` |
| `easy_preserved` | `True` |
| `all_min_without_group_positive` | `True` |
| `diagnostic_not_deployed` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
