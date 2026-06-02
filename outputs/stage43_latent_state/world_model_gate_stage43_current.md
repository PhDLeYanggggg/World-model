# Stage43 Current World-Model Gate

- source: `fresh_stage43_cd_source_family_coverage_guard`
- verdict: `stage43_cd_source_family_coverage_guard_pass`
- passed: `14 / 14`
- deployable policy changed: `False`
- test all improvement: `0.0111`
- test t50 improvement: `-0.0000`
- test hard/failure improvement: `0.0135`
- test easy degradation: `0.0000`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

## Current Boundary

- Stage43-CD is a source-family coverage guard audit for downstream latent heads.
- It does not remove or replace the current Stage37/Stage42 safety floor.
- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

| gate | passed |
| --- | --- |
| `stage43_cc_precondition_seen` | `True` |
| `fresh_coverage_replay_completed` | `True` |
| `train_only_heads_refit` | `True` |
| `validation_shadow_only_selection` | `True` |
| `coverage_guard_selected` | `True` |
| `unsupported_test_families_reported` | `True` |
| `shadow_holdout_easy_safe` | `True` |
| `test_easy_preserved` | `True` |
| `test_lift_vs_floor` | `True` |
| `test_t50_reported` | `True` |
| `test_hard_failure_reported` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
