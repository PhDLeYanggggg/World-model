# Frozen Stage42-HS Group-Consistency T100 Easy Guard Policy

- source: `cached_verified_stage42_hr_policy_freeze_from_fresh_artifact`
- base_stage: `Stage42-HR group-consistency t100 easy guard`
- selection_scope: `validation_only_domain_horizon_t100`
- deployment_role: `protected_t100_easy_guard_for_group_consistency_policy`
- decision_rule: `validation_all_gain > 0 and validation_easy_degradation <= threshold`
- fallback: `train-horizon causal floor for guarded domain|t100 slice`
- guarded_slices: `{'TrajNet|100': {'source': 'fresh_validation_only_domain_t100_easy_guard', 'domain': 'TrajNet', 'val_rows': 1160, 'test_rows': 5608, 'val_all_improvement': 0.23260462520508085, 'val_easy_degradation': 0.017118176622190173, 'threshold': 0.0, 'keep': False, 'reason': 'validation_easy_degradation_above_threshold_or_nonpositive_gain'}}`
- kept_slices: `{'UCY|100': {'source': 'fresh_validation_only_domain_t100_easy_guard', 'domain': 'UCY', 'val_rows': 1440, 'test_rows': 1440, 'val_all_improvement': 0.27564518723015075, 'val_easy_degradation': -0.021788147627511134, 'threshold': 0.0, 'keep': True}}`

Claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
