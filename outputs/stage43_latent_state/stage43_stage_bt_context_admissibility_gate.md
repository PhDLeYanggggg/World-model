# Stage43-BT Context Admissibility Gate

- verdict: `stage43_bt_context_admissibility_pass_safe_lift_diagnostic`
- passed: `14 / 14`
- row-level admissibility trained: `True`
- beats graph-history on any core metric: `True`
- easy safe: `True`
- deployable policy changed: `False`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | passed |
| --- | --- |
| `bp_precondition_passed` | `True` |
| `bq_precondition_passed` | `True` |
| `br_precondition_passed` | `True` |
| `bs_precondition_passed` | `True` |
| `fresh_torch_training_completed` | `True` |
| `checkpoint_not_committed` | `True` |
| `train_val_test_loaded` | `True` |
| `validation_only_threshold_selection` | `True` |
| `test_eval_completed` | `True` |
| `graph_history_reference_present` | `True` |
| `admissibility_diagnostics_reported` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
