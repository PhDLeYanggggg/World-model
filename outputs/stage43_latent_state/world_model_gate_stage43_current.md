# Stage43 Current World-Model Gate

- source: `fresh_stage43_cc_shadow_easy_guard_repair`
- verdict: `stage43_cc_shadow_easy_guard_shadow_safe_test_mismatch`
- passed: `12 / 13`
- deployable policy changed: `False`
- test all improvement: `0.0321`
- test t50 improvement: `-0.0083`
- test hard/failure improvement: `0.0655`
- test easy degradation: `0.0527`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

## Current Boundary

- Stage43-CC is a shadow-validation safety repair audit for downstream latent heads.
- It does not remove the Stage37/Stage42 safety floor.
- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

| gate | passed |
| --- | --- |
| `stage43_cb_precondition_seen` | `True` |
| `fresh_shadow_replay_completed` | `True` |
| `train_only_heads_refit` | `True` |
| `validation_split_internal_only` | `True` |
| `no_test_threshold_tuning` | `True` |
| `inference_safe_guard_features_only` | `True` |
| `shadow_holdout_easy_safe` | `True` |
| `test_easy_preserved` | `False` |
| `test_lift_vs_floor` | `True` |
| `source_family_support_reported` | `True` |
| `domain_horizon_source_breakdown_reported` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
