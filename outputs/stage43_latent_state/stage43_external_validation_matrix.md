# Stage43-AT External Validation Matrix

- source: `fresh_stage43_at_external_validation_matrix`
- result_source: `fresh_external_validation_matrix_from_verified_artifacts`
- verdict: `stage43_at_external_validation_matrix_pass`
- gate: `13 / 13`
- input hash: `f254174d4f826cfe24a496e71c253e18473a9fa0fe424f9cae1f7bbd3e201d75`
- split verdict: `stage43_f_source_level_split_ready`
- test rows: `89736`
- test domains: `['ETH_UCY', 'TrajNet', 'UCY']`
- test source count: `4`

## Comparison Matrix

| model / policy | role | source | deployable | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch | caveat |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `external strongest / Stage37 floor reference` | `safety_floor_reference` | `cached_verified_floor_reference` | `True` | 89736 | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | Reference floor. Improvements are reported relative to this floor where applicable. |
| `M3W-Neural v1 composite-tail safe-switch` | `previous_best_protected_neural` | `cached_verified` | `True` | 55528 | `21.03%` | `13.65%` | `14.69%` | `20.38%` | `0.00%` | `34.10%` | Earlier protected neural candidate under Stage37/teacher floor; not a new Stage43 source-level replay. |
| `Stage43-G source-level latent, ungated/full-switch diagnostic` | `ungated_neural_diagnostic` | `fresh_run_unit_consistency_and_safety_audit` | `False` | 89736 | `35.14%` | `15.81%` | `0.45%` | `37.74%` | `159.75%` | `100.00%` | Unit-consistent audit found easy degradation unsafe; keep floor. |
| `Stage43-I domain-capped protected latent safe-switch` | `protected_domain_level_neural` | `fresh_run_unit_consistent_safe_switch_repair` | `True` | 89736 | `23.11%` | `11.36%` | `1.35%` | `24.41%` | `0.00%` | `18.53%` | Domain-level positive and easy-safe, but one source slice remained slightly negative. |
| `Stage43-K validation source-family guarded repair` | `source_safe_protected_neural` | `fresh_run_source_slice_repair_without_test_threshold_tuning` | `True` | 89736 | `23.11%` | `11.36%` | `1.35%` | `24.41%` | `0.00%` | `18.52%` | Repairs negative source harm with validation-only source-family guard; does not claim every source has positive transfer. |
| `Stage43-M protected full-waypoint latent dynamics` | `protected_full_waypoint_neural` | `fresh_run` | `True` | 16000 | `37.23%` | `32.94%` | `-27.90%` | `38.77%` | `0.00%` | `89.76%` | Protected full-waypoint signal on 16k-row supervision cache; t100 remains guarded/diagnostic and not source-level official. |
| `Stage43-AO frozen bounded-residual replay` | `current_best_integrated_candidate` | `fresh_replay_from_frozen_policy_artifact` | `True` | 16000 | `38.00%` | `26.96%` | `0.00%` | `37.71%` | `0.00%` | `63.08%` | Exact reviewer replay of frozen bounded-residual policy; protected and h100 guarded, not a global floor removal. |
| `Stage43-P tail-horizon full-waypoint adapter` | `latest_full_test_tail_adapter_candidate` | `fresh_train_val_selected_tail_horizon_adapter` | `True` | 89736 | `50.25%` | `51.23%` | `0.00%` | `47.88%` | `0.00%` | `70.45%` | Latest full-test protected tail-horizon adapter; materially stronger on all/t50/hard, but h100 remains validation-blocked and falls back to the floor. |

## Per-Domain Protected External Validation

| domain | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | 70585 | `19.25%` | `0.16%` | `-0.72%` | `20.41%` | `0.95%` | `15.00%` |
| `TrajNet` | 9611 | `6.61%` | `0.00%` | `0.00%` | `7.47%` | `0.00%` | `10.00%` |
| `UCY` | 9540 | `53.57%` | `47.53%` | `15.70%` | `54.79%` | `0.00%` | `53.21%` |

## Source-Safe Repair Boundary

- negative source count after repair: `0`
- uniform positive per-source claim allowed: `False`
- reason: Unsupported or low-support source families may be safely floored to zero; that repairs harm but is not positive transfer.

## Claim Boundary

- This is a fresh Stage43 matrix assembled from verified artifacts; it is not a new threshold search.
- Dataset-local/raw-frame 2.5D only.
- No metric or seconds-level claim.
- No true 3D or foundation claim.
- No Stage5C execution and no SMC.
- Future endpoints/full waypoints remain label/eval only, not inference input.

## Gate

| gate | passed |
| --- | --- |
| `source_level_split_ready` | `True` |
| `external_domains_present` | `True` |
| `required_model_families_compared` | `True` |
| `ungated_unsafe_not_deployed` | `True` |
| `source_safe_candidate_present` | `True` |
| `uniform_source_overclaim_blocked` | `True` |
| `current_candidate_replay_exact_or_reconciled` | `True` |
| `latest_tail_adapter_candidate_present` | `True` |
| `per_domain_and_per_source_reported` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `global_floor_not_removed` | `True` |
| `long_objective_not_marked_complete` | `True` |
