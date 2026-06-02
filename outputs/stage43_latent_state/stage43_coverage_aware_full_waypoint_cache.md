# Stage43-CF Coverage-Aware Full-Waypoint Cache

- source: `fresh_stage43_cf_coverage_aware_full_waypoint_cache`
- result_source: `fresh_cache_rebuild_from_stage43_ce_assignment`
- verdict: `stage43_cf_coverage_aware_full_waypoint_cache_ready`
- gate: `14 / 14`
- cache dir: `data/stage43_ce_full_waypoint_supervision_cache`
- cache committed: `False`
- new model training run: `False`

## Purpose

- Stage43-CE produced a coverage-aware source assignment but did not rebuild training labels.
- Stage43-CF materializes the repaired full-waypoint supervision cache for train/val/test under that assignment.
- Future endpoint/full-waypoint data remain labels/evaluation targets only; they are not model inputs.

## Split Summary

| split | rows | domains | sources | scenes | horizons | full waypoint rows | all-waypoint rows | missing tracks | endpoint diff max | hard | failure | easy |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 192531 | `{'ETH_UCY': 108794, 'TrajNet': 36514, 'UCY': 47223}` | 12 | 7 | `{'10': 60567, '25': 53344, '50': 46176, '100': 32444}` | 192531 | 133352 | 0 | 0.00000000 | 147294 | 70447 | 59081 |
| val | 62796 | `{'ETH_UCY': 16103, 'TrajNet': 37153, 'UCY': 9540}` | 3 | 3 | `{'10': 19557, '25': 17499, '50': 15441, '100': 10299}` | 62796 | 43239 | 0 | 0.00000000 | 48498 | 26435 | 12396 |
| test | 82664 | `{'ETH_UCY': 25901, 'TrajNet': 47223, 'UCY': 9540}` | 3 | 3 | `{'10': 25430, '25': 22888, '50': 20345, '100': 14001}` | 82664 | 57234 | 0 | 0.00000000 | 59841 | 29605 | 25024 |

## Cache Files

| split | path | sha256 | row hash |
| --- | --- | --- | --- |
| train | `data/stage43_ce_full_waypoint_supervision_cache/stage43_ce_full_waypoint_supervision_train.npz` | `9c893895f5fdf26adb3a70540d6dc7c19272aa11f2d4193d3b3d68b43e67b8e1` | `cdcbbe5e1829d3f5c354d73e89f37bb902a76e29d7d4047ea0d6c14ad3f7221c` |
| val | `data/stage43_ce_full_waypoint_supervision_cache/stage43_ce_full_waypoint_supervision_val.npz` | `e234ff7da873d37b32e490b0f2fdba7983624af80a6066cf2fa95ad90f73e869` | `d7f83753a5c09a8130c06fb74949af7476157bac67ead57f2b17e48f9fe9291e` |
| test | `data/stage43_ce_full_waypoint_supervision_cache/stage43_ce_full_waypoint_supervision_test.npz` | `9922c10f91aea58ae297303a0ecc386a465e16b6f0c7551be529cefa3c9cc2b1` | `218f04e13ca83e3ee60a22d6917f66caa8b62981ed973e90d6d516c6a838a52f` |

## Leakage Boundary

- Source files are disjoint across train/val/test.
- No future endpoint/waypoint is used as an inference input.
- No central velocity, test endpoint goal construction, or test-statistics normalization is introduced.
- This cache is local derived data and is intentionally not committed.

## Claim Boundary

- This is a cache rebuild, not a new model result.
- Dataset-local/raw-frame 2.5D only.
- No metric/seconds, true-3D, foundation, Stage5C, or SMC claim.

## Next Required Step

- Train/evaluate the Stage43 full-waypoint latent dynamics model on this repaired cache.
- Keep the broad external stress matrix as diagnostic evidence; this repaired split is coverage-aware and narrower.

## Gate

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
