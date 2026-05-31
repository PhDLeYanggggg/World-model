# Stage43-L Full-Waypoint Supervision Cache

- source: `fresh_stage43_l_full_waypoint_supervision_cache`
- verdict: `stage43_l_full_waypoint_supervision_cache_pass`
- gate: `10 / 10`
- full-waypoint supervised training ready: `True`
- cache dir: `data/stage43_full_waypoint_supervision_cache`
- cache committed: `False`

## Why This Stage Exists

Stage43-B had endpoint/failure/gain/harm latent-state training ready, but full-waypoint supervised latent training was blocked because train/val/test full-waypoint labels were not frozen under the Stage43 source-level split. Stage43-L freezes those labels as a local cache and records row hashes, source disjointness, and leakage boundaries.

## Split Summary

| split | rows | domains | sources | scenes | horizons | full waypoint rows | all-waypoint rows | missing tracks | endpoint diff max | hard | failure | easy |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 146809 | `{'ETH_UCY': 63602, 'TrajNet': 73667, 'UCY': 9540}` | 10 | 5 | `{'10': 46716, '25': 41248, '50': 35467, '100': 23378}` | 146809 | 100093 | 0 | 0.00000000 | 112218 | 57390 | 37837 |
| val | 101446 | `{'ETH_UCY': 16611, 'TrajNet': 37612, 'UCY': 47223}` | 4 | 4 | `{'10': 32706, '25': 28703, '50': 24741, '100': 15296}` | 101446 | 68740 | 0 | 0.00000000 | 73296 | 35300 | 31737 |
| test | 89736 | `{'ETH_UCY': 70585, 'TrajNet': 9611, 'UCY': 9540}` | 4 | 4 | `{'10': 26132, '25': 23780, '50': 21754, '100': 18070}` | 89736 | 64992 | 0 | 0.00000000 | 70119 | 33797 | 26927 |

## Cache Files

| split | path | sha256 | row hash |
| --- | --- | --- | --- |
| train | `data/stage43_full_waypoint_supervision_cache/stage43_full_waypoint_supervision_train.npz` | `7cb12a873447bc9f59d135ced4db4dd4aeeea43257982c397596fb29e90b63e8` | `7a9ed6e07f1199a5791f035a463d14792959293ecab9c8a6f747f6f9a57cc076` |
| val | `data/stage43_full_waypoint_supervision_cache/stage43_full_waypoint_supervision_val.npz` | `ec68b9f3fda0166f918f18bda124262cf974829e752f89770a489f14f099df5d` | `d5160c4cbcff9c2fdebd53b8171b9955d48729ec0ee31433f0b78d8dc5b736bf` |
| test | `data/stage43_full_waypoint_supervision_cache/stage43_full_waypoint_supervision_test.npz` | `62f4a079bfa7e3eba5ca5ebdadf864d7a1342b4dc4431412ecaed910520ab22f` | `18f6ded3c61645488becef8f842401669037776875508686a37bd9da74544a5b` |

## Leakage Boundary

- Future waypoints and endpoints are labels/evaluation targets only.
- They are not inference inputs.
- Source files are disjoint across train/val/test.
- No test endpoint goal construction, no central velocity input, no test-statistics normalization.

## Gate

| gate | passed |
| --- | --- |
| stage43_source_split_precondition_passed | True |
| cache_files_written | True |
| train_val_test_rows_present | True |
| train_val_full_waypoint_labels_present | True |
| test_full_waypoint_labels_present | True |
| endpoint_alignment_pass | True |
| source_splits_disjoint | True |
| no_future_waypoint_input | True |
| no_test_goal_or_stat_leakage | True |
| no_metric_seconds_stage5c_smc_claim | True |

Conclusion: Stage43-L closes the Stage43-B full-waypoint supervision cache blocker for local source-level raw-frame training. It does not execute Stage5C or SMC and does not make metric/seconds/true-3D/foundation claims.
