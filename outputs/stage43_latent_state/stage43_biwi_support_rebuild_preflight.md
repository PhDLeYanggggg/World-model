# Stage43-BD Biwi Support Rebuild Preflight

- source: `fresh_stage43_bd_biwi_support_rebuild_preflight`
- result_source: `fresh_biwi_source_family_support_rebuild_preflight`
- verdict: `stage43_bd_biwi_support_rebuild_preflight_pass`
- gate: `14 / 14`
- biwi sources in feature store: `2`
- current train / val / test rows: `0 / 459 / 7685`
- deployable repair options now: `0`

## Current Biwi Sources

| source | raw role | rows | t50 rows | old splits | current source-level split |
| --- | --- | ---: | ---: | --- | --- |
| `biwi_eth.txt` | `raw_test_dir` | 459 | 51 | `{'train': 459}` | `val` |
| `biwi_hotel.txt` | `raw_train_dir` | 7685 | 1885 | `{'train': 7685}` | `test` |

## Rebuild Options

| option | split type | rows train/val/test | t50 train/val/test | deployable repair allowed | blockers |
| --- | --- | --- | --- | --- | --- |
| `raw_train_support_raw_test_validation` | `family_support_candidate_not_deployable` | 7685/459/0 | 1885/51/0 | `False` | no_independent_biwi_test_source_after_support_rebuild, current_stage43_test_source_would_move_to_train_support, validation_rows_below_threshold, validation_t50_rows_below_threshold |
| `within_source_agent_split_support_diagnostic` | `diagnostic_within_source_cv_not_official_source_level` | 6254/1431/0 | 1534/351/0 | `False` | within_source_split_not_source_level_heldout, no_independent_biwi_test_source, current_stage43_test_source_would_be_reused_for_support |
| `keep_current_stage43_floor_only` | `deployable_current_floor` | 0/0/8144 | 0/0/1936 | `False` | no_train_family_rows_for_repair |

## Interpretation

This preflight makes the next boundary explicit. `biwi_hotel` has enough rows to build support, but it is also the current Stage43 held-out biwi test source. Moving it into training would invalidate the current source-level test claim. `biwi_eth` is small and comes from the raw Test directory, so I do not use it for training.

The only safe conclusion is still conservative: there is support to test a diagnostic converter, but not enough independent source-level evidence to train and deploy a biwi repair. Stage43-P/AZ should keep the floor on `TrajNet_biwi` until another independent biwi-like source or a new source-level protocol is available.

## Next Required Actions

- Do not train a biwi-specific repair on the current Stage43 source-level test source.
- If biwi repair remains important, acquire or locate an independent biwi-like source so train, validation, and test support are disjoint.
- A within-source agent split can be used only as a diagnostic conversion smoke test, not as deployable evidence.
- Keep Stage43-P/AZ floor behavior for TrajNet_biwi until a source-level validation gate has enough rows and positive easy-safe evidence.

## Claim Boundary

- This is a preflight manifest, not a model result.
- Dataset-local/raw-frame 2.5D only.
- No metric or seconds-level claim.
- No biwi deployable repair claim.
- No Stage5C execution and no SMC.

## Gate

| gate | passed |
| --- | --- |
| `stage43_bc_precondition_passed` | `True` |
| `stage43_source_split_precondition_passed` | `True` |
| `biwi_sources_found_in_feature_store` | `True` |
| `current_train_gap_reconfirmed` | `True` |
| `candidate_rebuild_options_evaluated` | `True` |
| `deployable_repair_correctly_blocked` | `True` |
| `diagnostic_support_option_recorded` | `True` |
| `raw_test_training_blocked` | `True` |
| `current_test_reuse_blocked_for_deployment` | `True` |
| `next_actions_recorded` | `True` |
| `no_future_or_test_leakage` | `True` |
| `claim_boundary_not_overstated` | `True` |
| `stage5c_and_smc_false` | `True` |
| `long_objective_kept_active` | `True` |
