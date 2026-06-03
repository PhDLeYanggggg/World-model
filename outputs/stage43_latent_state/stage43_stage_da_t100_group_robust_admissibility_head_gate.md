# Stage43-DA Gate

- verdict: `stage43_da_t100_group_robust_head_positive_but_not_policy_best`
- passed: `14 / 14`
- beats CZ t100 mean: `False`
- deploy on current heldout t100: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass |
| --- | --- |
| `stage43_cz_precondition_present` | `True` |
| `fresh_group_robust_training` | `True` |
| `three_or_more_seeds` | `True` |
| `all_checkpoints_written_not_committed` | `True` |
| `feature_contract_clean` | `True` |
| `group_weighting_used` | `True` |
| `leave_group_out_validation_selection` | `True` |
| `test_once_per_seed` | `True` |
| `easy_preserved` | `True` |
| `group_fragility_measured` | `True` |
| `diagnostic_not_deployed` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
