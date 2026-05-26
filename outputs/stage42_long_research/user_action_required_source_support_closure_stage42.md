# Stage42-DD User Action Required For Source Support Closure

- source: `fresh_stage42_dd_source_support_closure_audit`
- purpose: close DA-1 blockers for legal/source/time-calibrated ETH_UCY, TrajNet, and UCY evidence.

## ETH_UCY

- blockers: `['source_terms_confirmation_or_conversion_readiness_missing', 'train_only_t100_source_cv_support_missing', 'additional_t100_sources_needed=2', 'legal_terms_blocked_targets=eth_biwi_original']`
- partial_support: `{'source_specific_metric_time_sources': ['ETH_seq_eth', 'ETH_seq_hotel'], 'local_t100_schema_source_cv_evaluated': True, 'local_t100_schema_positive_vs_constant_velocity': False, 'preflight_targets_with_t50_files': 2, 'preflight_targets_with_t100_files': 2}`
- action: confirm ETH/BIWI or ETH-Person source terms and add enough independent t100-capable ETH_UCY train sources, then rerun source-CV without test tuning

## TrajNet

- blockers: `['source_terms_confirmation_or_conversion_readiness_missing', 'train_only_t100_source_cv_support_missing', 'additional_t100_sources_needed=1', 'source_specific_metric_time_calibration_missing', 'legal_terms_blocked_targets=trajnetplusplus_official']`
- partial_support: `{'source_specific_metric_time_sources': [], 'local_t100_schema_source_cv_evaluated': False, 'local_t100_schema_positive_vs_constant_velocity': False, 'preflight_targets_with_t50_files': 1, 'preflight_targets_with_t100_files': 1}`
- action: provide/confirm legal TrajNet++ or TrajNet-compatible long-track source with timing/geometry evidence, then rerun conversion, no-leakage, and train-only source-CV

## UCY

- blockers: `['source_terms_confirmation_or_conversion_readiness_missing', 'train_only_t100_source_cv_support_missing', 'additional_t100_sources_needed=1', 'legal_terms_blocked_targets=ucy_crowd_original']`
- partial_support: `{'source_specific_metric_time_sources': ['UCY_zara01', 'UCY_zara02', 'UCY_zara03', 'UCY_students03', 'UCY_students01'], 'local_t100_schema_source_cv_evaluated': True, 'local_t100_schema_positive_vs_constant_velocity': True, 'preflight_targets_with_t50_files': 2, 'preflight_targets_with_t100_files': 2}`
- action: confirm UCY original terms/source identity and add one independent t100-capable UCY source or source split before claiming stable t100

## Non-Claims

- Do not claim global metric or seconds-level M3W results from these blockers.
- Do not claim global t100 deployable success while train-only source-CV support is missing.
- Do not treat local path existence, parseability, or OpenTraj toolkit license as underlying dataset permission.
