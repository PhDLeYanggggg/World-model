# Stage43-BU Context Admissibility Robustness Audit

- source: `fresh_stage43_bu_context_admissibility_robustness_audit`
- result_source: `fresh_replay_bootstrap_slice_audit_from_stage43_bt`
- verdict: `stage43_bu_context_admissibility_partial_robust_lift_pass`
- gate: `12 / 12`
- robust all/hard lift: `True`
- t50 bootstrap robust: `True`
- t100 bootstrap robust: `False`
- t100 CI crosses zero: `True`
- slice easy safe: `False`
- easy-safe CI: `True`
- deployable policy changed: `False`

## Exact Replay

- checkpoint committed: `False`
- replay diff max: `0.00000000`
- replay diff: `{'full_waypoint_ade_improvement_vs_floor': 0.0, 't50_full_waypoint_ade_improvement_vs_floor': 0.0, 't100_raw_frame_full_waypoint_diagnostic_vs_floor': 0.0, 'hard_failure_full_waypoint_ade_improvement_vs_floor': 0.0, 'easy_degradation_vs_floor': 0.0}`

## Replay Metrics

- all full-waypoint ADE improvement: `39.06%`
- t50 full-waypoint ADE improvement: `16.02%`
- t100 raw-frame diagnostic improvement: `-3.21%`
- hard/failure improvement: `39.66%`
- easy degradation: `0.00%`
- switch rate: `57.44%`

## Delta Vs Graph-History-Only

- all delta: `2.15%`
- t50 delta: `0.40%`
- t100 raw-frame diagnostic delta: `0.05%`
- hard/failure delta: `2.02%`
- easy degradation delta: `0.00%`

## Bootstrap Delta Vs Graph-History-Only

- bootstrap n: `2000`

| metric | rows | low | mean | high |
| --- | ---: | ---: | ---: | ---: |
| `all_delta_vs_graph` | `12000` | `1.91%` | `2.15%` | `2.39%` |
| `t50_delta_vs_graph` | `2888` | `0.00%` | `0.40%` | `0.79%` |
| `t100_raw_frame_delta_vs_graph` | `2406` | `-0.11%` | `0.05%` | `0.21%` |
| `hard_failure_delta_vs_graph` | `9371` | `1.78%` | `2.02%` | `2.25%` |
| `easy_degradation_delta_vs_graph` | `3607` | `0.00%` | `0.00%` | `0.00%` |

## Slice Audit

- slice count: `39`
- positive slice count: `31`
- negative slice count: `2`
- easy hazard slice count: `7`
- core weak slices: `[]`

### Top Positive Slices

| slice | rows | delta vs graph | context rate | easy degradation |
| --- | ---: | ---: | ---: | ---: |
| `source_/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/biwi/biwi_hotel.txt_horizon_10` | `338` | `10.32%` | `33.73%` | `45.76%` |
| `domain_TrajNet_horizon_10` | `516` | `8.47%` | `42.83%` | `81.88%` |
| `domain_UCY_horizon_10` | `388` | `6.47%` | `37.11%` | `0.00%` |
| `source_/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt_horizon_10` | `388` | `6.47%` | `37.11%` | `0.00%` |
| `horizon_10` | `3544` | `5.16%` | `34.00%` | `0.00%` |
| `domain_ETH_UCY_horizon_10` | `2640` | `4.49%` | `31.82%` | `0.00%` |
| `source_/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt_horizon_10` | `2640` | `4.49%` | `31.82%` | `0.00%` |
| `domain_UCY` | `1271` | `4.10%` | `25.33%` | `0.00%` |
| `source_/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt` | `1271` | `4.10%` | `25.33%` | `0.00%` |
| `domain_UCY_horizon_25` | `366` | `3.95%` | `36.61%` | `0.00%` |
| `source_/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt_horizon_25` | `366` | `3.95%` | `36.61%` | `0.00%` |
| `source_/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/mot/PETS09-S2L1.txt` | `247` | `3.78%` | `50.20%` | `111.69%` |
| `...` | `8 more` |  |  |  |

### Top Negative Slices

| slice | rows | delta vs graph | context rate | easy degradation |
| --- | ---: | ---: | ---: | ---: |
| `domain_ETH_UCY_horizon_50` | `2333` | `-0.02%` | `3.39%` | `0.00%` |
| `source_/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt_horizon_50` | `2333` | `-0.02%` | `3.39%` | `0.00%` |

### Top Easy Hazard Slices

| slice | rows | delta vs graph | context rate | easy degradation |
| --- | ---: | ---: | ---: | ---: |
| `source_/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/mot/PETS09-S2L1.txt_horizon_10` | `178` | `2.80%` | `60.11%` | `122.34%` |
| `source_/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/mot/PETS09-S2L1.txt` | `247` | `3.78%` | `50.20%` | `111.69%` |
| `domain_TrajNet_horizon_10` | `516` | `8.47%` | `42.83%` | `81.88%` |
| `source_/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/biwi/biwi_hotel.txt_horizon_10` | `338` | `10.32%` | `33.73%` | `45.76%` |
| `domain_TrajNet` | `1255` | `3.61%` | `19.44%` | `18.20%` |
| `domain_TrajNet_horizon_50` | `249` | `0.00%` | `0.40%` | `2.44%` |
| `source_/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/biwi/biwi_hotel.txt_horizon_50` | `249` | `0.00%` | `0.40%` | `2.44%` |

## Interpretation

- Stage43-BU exact-replays Stage43-BT and adds bootstrap plus source/domain/horizon slice evidence.
- It is a robustness audit, not a deployment policy update.
- The key question is whether BT's row-level context admissibility lift is stable enough to claim context contribution.
- Dataset-local/raw-frame 2.5D only; t100 remains raw-frame diagnostic; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `bt_precondition_passed` | `True` |
| `checkpoint_replayed_not_committed` | `True` |
| `exact_replay_matches_bt_report` | `True` |
| `bootstrap_completed` | `True` |
| `slice_audit_completed` | `True` |
| `slice_easy_hazards_reported` | `True` |
| `all_and_hard_bootstrap_measured` | `True` |
| `easy_safety_ci_measured` | `True` |
| `t50_and_t100_reported` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
