# Stage43-F Source-Level Heldout Split

- source: `fresh_stage43_f_source_level_heldout_split`
- verdict: `stage43_f_source_level_split_ready`
- gate: `11 / 11`
- split granularity: `source_file_level`
- row hash: `9c8b4d51e0f7a1618dce410c7dd23fbf7f21da5de587d4ae021257775164c3c5`

## Pool

- rows: `337991`
- domains: `['ETH_UCY', 'TrajNet', 'UCY']`
- domain counts: `{'ETH_UCY': 150798, 'TrajNet': 120890, 'UCY': 66303}`
- source count: `18`
- scene count: `11`
- horizon counts: `{'10': 105554, '25': 93731, '50': 81962, '100': 56744}`

## New Source-Level Split

| split | rows | domains | sources | scenes | horizons | hard | failure | easy |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| train | 146809 | `['ETH_UCY', 'TrajNet', 'UCY']` | 10 | 5 | `{'10': 46716, '25': 41248, '50': 35467, '100': 23378}` | 112218 | 57390 | 37837 |
| val | 101446 | `['ETH_UCY', 'TrajNet', 'UCY']` | 4 | 4 | `{'10': 32706, '25': 28703, '50': 24741, '100': 15296}` | 73296 | 35300 | 31737 |
| test | 89736 | `['ETH_UCY', 'TrajNet', 'UCY']` | 4 | 4 | `{'10': 26132, '25': 23780, '50': 21754, '100': 18070}` | 70119 | 33797 | 26927 |

## Leakage Boundary

- source files disjoint: `True`
- row overlap pass: `True`
- scene overlap counts: `{'train_val': 1, 'train_test': 0, 'val_test': 1}`
- scene overlap is reported rather than hidden because this is a source-file-level split, not a strict scene-level split.
- no future endpoint/waypoint input, central velocity input, test endpoint goals, or test statistics normalization is constructed by this manifest.

## Claim Boundary

- This is a fresh split manifest, not a new model training/evaluation result.
- The old Stage43-C checkpoint remains UCY-heldout evidence only and is not official for this new source-level split.
- Coordinates remain dataset-local/raw-frame; no metric or seconds-level claim is made.
- Stage5C and SMC remain disabled.

## Gate

| gate | passed |
| --- | --- |
| input_pool_loaded | True |
| all_required_domains_present | True |
| source_level_split_built | True |
| train_val_test_nonempty | True |
| each_domain_has_test_and_train_sources | True |
| test_contains_all_domains | True |
| source_file_disjoint | True |
| row_overlap_pass | True |
| no_future_or_test_leakage_constructed | True |
| old_split_reuse_boundary_recorded | True |
| no_metric_seconds_stage5c_smc_claim | True |
