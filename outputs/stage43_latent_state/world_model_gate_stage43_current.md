# Stage43 Current World-Model Gate

- source: `fresh_stage43_cb_downstream_easy_guard_audit`
- verdict: `stage43_cb_downstream_easy_guard_val_safe_test_easy_mismatch`
- passed: `12 / 13`
- deployable policy changed: `False`
- test all improvement: `0.0321`
- test t50 improvement: `-0.0083`
- test hard/failure improvement: `0.0656`
- test easy degradation: `0.0528`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

## Current Boundary

- Stage43-CB is an easy-safety transfer audit for downstream latent heads.
- It does not change the deployable model when test easy preservation fails.
- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

| gate | passed |
| --- | --- |
| `stage43_ca_precondition_seen` | `True` |
| `fresh_guard_replay_completed` | `True` |
| `train_only_heads_refit` | `True` |
| `future_labels_eval_only` | `True` |
| `no_test_threshold_tuning` | `True` |
| `inference_safe_guard_features_only` | `True` |
| `validation_easy_safe_policy_found` | `True` |
| `test_easy_preserved` | `False` |
| `protected_lift_vs_floor` | `True` |
| `validation_test_easy_gap_reported` | `True` |
| `domain_horizon_source_breakdown_reported` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
