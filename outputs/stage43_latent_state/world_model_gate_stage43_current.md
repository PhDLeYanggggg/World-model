# Stage43 Current World-Model Gate

- source: `fresh_stage43_bd_biwi_support_rebuild_preflight`
- verdict: `stage43_bd_biwi_support_rebuild_preflight_pass`
- passed: `14 / 14`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

## Current Boundary

- Stage43-P / AZ remains the performance leader and exact replay artifact.
- Stage43-BC found raw biwi support, but Stage43-BD shows it is not yet deployable repair support.
- TrajNet_biwi stays floor-only until independent source-level support exists.

| gate | passed |
| --- | --- |
| stage43_bc_precondition_passed | `True` |
| stage43_source_split_precondition_passed | `True` |
| biwi_sources_found_in_feature_store | `True` |
| current_train_gap_reconfirmed | `True` |
| candidate_rebuild_options_evaluated | `True` |
| deployable_repair_correctly_blocked | `True` |
| diagnostic_support_option_recorded | `True` |
| raw_test_training_blocked | `True` |
| current_test_reuse_blocked_for_deployment | `True` |
| next_actions_recorded | `True` |
| no_future_or_test_leakage | `True` |
| claim_boundary_not_overstated | `True` |
| stage5c_and_smc_false | `True` |
| long_objective_kept_active | `True` |
