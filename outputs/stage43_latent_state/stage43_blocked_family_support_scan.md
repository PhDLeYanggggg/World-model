# Stage43-BC Blocked Family Support Scan

- source: `fresh_stage43_bc_blocked_family_support_scan`
- result_source: `fresh_raw_external_scan_for_blocked_source_family_support`
- verdict: `stage43_bc_blocked_family_support_scan_pass`
- gate: `12 / 12`
- raw files scanned: `59`
- parseable raw files: `38`
- blocked families: `2`
- repair training allowed now: `0`

## Blocked Family Actions

| family | raw files | raw train | raw test | raw t50 windows | current train rows | current val rows | recommendation | blockers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `TrajNet_biwi` | 2 | 1 | 1 | 2175 | 0 | 459 | `rebuild_source_family_split_with_raw_candidate_support_before_any_repair_training` | current_feature_store_has_no_train_family_rows, current_validation_support_below_threshold, existing_ungated_transfer_catastrophic_negative |
| `TrajNet_mot` | 1 | 1 | 0 | 0 | 0 | 0 | `acquire_additional_source_family_data_before_repair_training` | current_feature_store_has_no_train_family_rows, current_validation_support_below_threshold, raw_scan_has_no_t50_candidate_windows, existing_ungated_transfer_catastrophic_negative, single_source_family_no_independent_support_file |

## Family Raw Summary

| family | files | rows | tracks | t50 diagnostic windows | t100 diagnostic windows | roles |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `TrajNet_biwi` | 2 | 2900 | 145 | 2175 | 1450 | raw_test_dir:1, raw_train_dir:1 |
| `TrajNet_crowds` | 8 | 44220 | 2211 | 33165 | 22110 | raw_test_dir:3, raw_train_dir:5 |
| `TrajNet_mot` | 1 | 2140 | 107 | 0 | 0 | raw_train_dir:1 |
| `other` | 48 | 179700 | 8985 | 0 | 0 | raw_test_dir:17, raw_train_dir:31 |

## Interpretation

This scan is a data-support step, not a new model result. It says `TrajNet_biwi` has raw candidate material that could be converted into support, but the current feature-store split still has no train rows for that family. `TrajNet_mot` remains a harder blocker: the raw scan finds only the current PETS source and no independent family support.

So the next legitimate move is a guarded conversion/split rebuild, not source repair training. Test rows remain diagnostic and are not used for thresholds or training.

## Next Required Actions

- For TrajNet_biwi, rebuild a legal source-family support split using raw biwi candidates before repair training.
- For TrajNet_mot, acquire or locate another independent MOT-like source; the current scan finds no independent validation support.
- After any support conversion, rerun no-leakage and validation-only support gates before evaluating test.
- Keep Stage43-P/AZ floor-only behavior on blocked sources until support gates clear.

## Claim Boundary

- Dataset-local/raw-frame 2.5D only.
- Horizon window counts here are diagnostic raw-file availability, not official model metrics.
- No metric or seconds-level claim.
- No true 3D or foundation claim.
- No Stage5C execution and no SMC.

## Gate

| gate | passed |
| --- | --- |
| `stage43_bb_precondition_passed` | `True` |
| `raw_external_sources_scanned` | `True` |
| `raw_parseability_reported` | `True` |
| `blocked_families_have_actions` | `True` |
| `support_candidates_separated_from_training_permission` | `True` |
| `mot_blocker_recorded` | `True` |
| `biwi_support_candidate_recorded` | `True` |
| `next_actions_recorded` | `True` |
| `no_future_or_test_leakage` | `True` |
| `claim_boundary_not_overstated` | `True` |
| `stage5c_and_smc_false` | `True` |
| `long_objective_kept_active` | `True` |
