# Stage43 Current World-Model Gate

- source: `fresh_stage43_bg_blocked_source_terms_validator`
- verdict: `stage43_bg_blocked_source_terms_validation_pass`
- passed: `13 / 13`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

## Current Boundary

- Stage43-P / AZ remains the performance leader and exact replay artifact.
- Stage43-BG validates the blocked-source terms template; it does not accept terms, convert data, or train.
- PETS, Town-Center, Wild-Track, and biwi stay floor-only until source/terms/split/conversion gates clear.

| gate | passed |
| --- | --- |
| stage43_bf_precondition_passed | `True` |
| template_loaded | `True` |
| datasets_validated | `True` |
| all_rows_have_blocker_status | `True` |
| blank_template_blocks_conversion | `True` |
| manifest_written | `True` |
| conversion_training_eval_zero | `True` |
| biwi_repair_still_blocked | `True` |
| no_future_or_test_leakage | `True` |
| no_execution | `True` |
| claim_boundary_not_overstated | `True` |
| stage5c_and_smc_false | `True` |
| long_objective_kept_active | `True` |
