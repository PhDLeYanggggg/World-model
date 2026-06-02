# Stage43 Current World-Model Gate

- source: `fresh_stage43_cf_coverage_aware_full_waypoint_cache`
- verdict: `stage43_cf_coverage_aware_full_waypoint_cache_ready`
- passed: `14 / 14`
- deployable policy changed: `False`
- new model training run: `False`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

## Current Boundary

- Stage43-CF rebuilds the full-waypoint supervision cache under the CE coverage-aware split.
- It is not a model result and does not replace the current safety floor.
- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

| gate | passed |
| --- | --- |
| `stage43_ce_precondition_ready` | `True` |
| `cache_files_written` | `True` |
| `cache_rows_match_ce_assignment` | `True` |
| `train_val_test_rows_present` | `True` |
| `full_waypoint_labels_present` | `True` |
| `endpoint_alignment_pass` | `True` |
| `source_splits_disjoint` | `True` |
| `validation_covers_test_source_families` | `True` |
| `no_future_waypoint_input` | `True` |
| `no_test_goal_or_stat_leakage` | `True` |
| `cache_not_committed_boundary` | `True` |
| `not_a_model_result_boundary_recorded` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
