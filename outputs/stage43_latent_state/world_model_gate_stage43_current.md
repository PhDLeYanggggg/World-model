# Stage43 Current World-Model Gate

- source: `fresh_stage43_bb_blocked_source_repair_feasibility`
- verdict: `stage43_bb_blocked_source_repair_feasibility_pass`
- passed: `12 / 12`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

## Current Boundary

- Stage43-P / AZ remains the performance leader and exact replay artifact.
- Stage43-BB says the blocked sources are not repairable safely yet.
- TrajNet_biwi and TrajNet_mot remain floor-only until validation support improves.

| gate | passed |
| --- | --- |
| stage43_ba_precondition_passed | `True` |
| blocked_sources_inspected | `True` |
| split_support_quantified | `True` |
| validation_support_quantified | `True` |
| unsafe_repair_correctly_blocked | `True` |
| catastrophic_ungated_transfer_not_deployed | `True` |
| diagnostic_test_not_used_for_training | `True` |
| next_actions_recorded | `True` |
| no_future_or_test_leakage | `True` |
| claim_boundary_not_overstated | `True` |
| stage5c_and_smc_false | `True` |
| long_objective_kept_active | `True` |
