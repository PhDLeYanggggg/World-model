# Stage43-CR Gate

- verdict: `stage43_cr_t100_supported_latent_dynamics_keep_floor`
- passed: `14 / 14`
- deploy on current heldout t100: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass |
| --- | --- |
| `stage43_cq_precondition_passed` | `True` |
| `torch_training_fresh_run` | `True` |
| `checkpoint_written_not_committed` | `True` |
| `t100_only_train_val_test` | `True` |
| `feature_contract_clean` | `True` |
| `latent_noncollapse` | `True` |
| `validation_selected_policy` | `True` |
| `test_once_completed` | `True` |
| `easy_preserved` | `True` |
| `protected_lift_or_honest_floor` | `True` |
| `current_heldout_t100_not_changed` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
