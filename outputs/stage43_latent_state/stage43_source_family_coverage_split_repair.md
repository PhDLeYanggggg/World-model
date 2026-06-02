# Stage43-CE Source-Family Coverage Split Repair

- source: `fresh_stage43_ce_source_family_coverage_split_repair`
- result_source: `fresh_metadata_only_source_family_coverage_split_repair`
- verdict: `stage43_ce_source_family_coverage_split_repair_ready`
- gate: `14 / 14`
- assignment hash: `4df1b5fd2309b0b5fffdd86aabd1445c3359c254c3c2f80a871b671f4f4b2c82`
- deployable policy changed: `False`
- new model training run: `False`

## Why This Exists

- Stage43-CD proved that validation source-family coverage gaps make the downstream latent guard overly conservative.
- This audit builds a metadata-only source split where validation covers every test source family and domain-family when feasible.
- It does not use test labels, future endpoints, future waypoints, or test metrics for threshold tuning.

## Coverage-Aware Split

| split | rows | domains | families | domain families | sources |
| --- | ---: | --- | --- | --- | ---: |
| train | 192531 | `['ETH_UCY', 'TrajNet', 'UCY']` | `['biwi', 'crowds', 'hotel', 'obsmat', 'pets', 'students', 'zara']` | `['ETH_UCY|hotel', 'ETH_UCY|obsmat', 'ETH_UCY|students', 'TrajNet|biwi', 'TrajNet|crowds', 'TrajNet|pets', 'TrajNet|students', 'TrajNet|zara', 'UCY|students']` | 12 |
| val | 62796 | `['ETH_UCY', 'TrajNet', 'UCY']` | `['students', 'zara']` | `['ETH_UCY|zara', 'TrajNet|students', 'UCY|zara']` | 3 |
| test | 82664 | `['ETH_UCY', 'TrajNet', 'UCY']` | `['students', 'zara']` | `['ETH_UCY|zara', 'TrajNet|students', 'UCY|zara']` | 3 |

## Validation Coverage

- test families without validation support: `[]`
- test domain-families without validation support: `[]`
- singleton domain-families avoided in test: `True`
- tradeoff: the repaired test split is coverage-aware and narrower than the broad external stress split; unsupported singleton families remain acquisition/coverage blockers rather than hidden successes.

## Leakage / Caveats

- source-file disjoint: `True`
- basename overlap counts: `{'train_val': 1, 'train_test': 1, 'val_test': 2}`
- scene overlap counts: `{'train_val': 1, 'train_test': 1, 'val_test': 1}`
- Scene overlap is reported because this is still source-file-level, not strict scene-level.
- Basename overlap is reported as a duplicate-source caution, not hidden.

## Next Required Step

- Rebuild the Stage43 full-waypoint supervision cache with this assignment, then retrain/evaluate latent models on that repaired split.
- Until that happens, this is split-repair readiness evidence only, not a new world-model result.
- Keep the broad external stress matrix as a separate diagnostic so the repaired split does not erase domain-gap evidence.

## Claim Boundary

- Dataset-local/raw-frame 2.5D only.
- No metric/seconds, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `stage43_f_precondition_passed` | `True` |
| `coverage_aware_assignment_built` | `True` |
| `train_val_test_nonempty` | `True` |
| `test_contains_required_domains` | `True` |
| `validation_contains_required_domains` | `True` |
| `source_file_disjoint` | `True` |
| `global_source_family_validation_coverage` | `True` |
| `domain_source_family_validation_coverage` | `True` |
| `singleton_unsupported_families_avoided_in_test` | `True` |
| `basename_overlap_reported` | `True` |
| `no_future_or_test_leakage_constructed` | `True` |
| `not_a_model_result_boundary_recorded` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
