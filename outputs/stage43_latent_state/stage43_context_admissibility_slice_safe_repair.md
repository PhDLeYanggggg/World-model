# Stage43-BV Context Admissibility Slice-Safe Repair

- source: `fresh_stage43_bv_context_admissibility_slice_safe_repair`
- result_source: `fresh_validation_selected_slice_safe_context_repair`
- verdict: `stage43_bv_context_admissibility_slice_repair_diagnostic_remaining_risk`
- gate: `12 / 12`
- selected repair mode: `block_t100`
- safe validation candidates: `0 / 10`
- easy safe: `True`
- slice easy safe: `False`
- t100 bootstrap robust: `False`
- deployable policy changed: `False`

## Test Metrics

- all full-waypoint ADE improvement: `39.06%`
- t50 full-waypoint ADE improvement: `16.02%`
- t100 raw-frame diagnostic improvement: `-3.26%`
- hard/failure improvement: `39.65%`
- easy degradation: `0.00%`
- switch rate: `57.51%`
- context counts on test: `{'graph_history_only': 10344, 'scene_graph_full': 527, 'scene_proxy_only': 1129}`
- blocked context rows on test: `22`

## Delta Vs Graph-History-Only

- all delta: `2.14%`
- t50 delta: `0.40%`
- t100 raw-frame diagnostic delta: `0.00%`
- hard/failure delta: `2.01%`
- easy degradation delta: `0.00%`

## Validation Candidate Repair Policies

| mode | safe | score | all delta | t50 delta | t100 delta | hard delta | easy delta | hazards | context rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `block_t100` | `False` | `0.02460` | `3.57%` | `0.28%` | `0.00%` | `2.76%` | `0.00%` | `4` | `20.46%` |
| `block_unsafe_horizon` | `False` | `0.02460` | `3.57%` | `0.28%` | `0.00%` | `2.76%` | `0.00%` | `4` | `20.46%` |
| `require_domain_or_source_safe` | `False` | `0.02460` | `3.57%` | `0.28%` | `0.00%` | `2.76%` | `0.00%` | `4` | `20.46%` |
| `bt_unrepaired` | `False` | `0.02451` | `3.56%` | `0.28%` | `-0.01%` | `2.76%` | `0.00%` | `4` | `20.47%` |
| `block_unsafe_domain_horizon` | `False` | `0.01932` | `3.31%` | `0.28%` | `0.00%` | `2.47%` | `0.00%` | `4` | `18.61%` |
| `block_unsafe_source_horizon` | `False` | `0.01932` | `3.31%` | `0.28%` | `0.00%` | `2.47%` | `0.00%` | `4` | `18.61%` |
| `hierarchical_any_unsafe` | `False` | `0.01932` | `3.31%` | `0.28%` | `0.00%` | `2.47%` | `0.00%` | `4` | `18.61%` |
| `source_or_domain_safe_and_horizon_safe` | `False` | `0.01932` | `3.31%` | `0.28%` | `0.00%` | `2.47%` | `0.00%` | `4` | `18.61%` |
| `strict_safe_no_t100` | `False` | `0.01932` | `3.31%` | `0.28%` | `0.00%` | `2.47%` | `0.00%` | `4` | `18.61%` |
| `all_fallback` | `False` | `-0.06000` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `0.00%` | `6` | `0.00%` |

## Bootstrap Delta Vs Graph-History-Only

- bootstrap n: `2000`

| metric | rows | low | mean | high |
| --- | ---: | ---: | ---: | ---: |
| `all_delta_vs_graph` | `12000` | `1.91%` | `2.14%` | `2.38%` |
| `t50_delta_vs_graph` | `2888` | `-0.02%` | `0.39%` | `0.77%` |
| `t100_raw_frame_delta_vs_graph` | `2406` | `0.00%` | `0.00%` | `0.00%` |
| `hard_failure_delta_vs_graph` | `9371` | `1.79%` | `2.01%` | `2.23%` |
| `easy_degradation_delta_vs_graph` | `3607` | `0.00%` | `0.00%` | `0.00%` |

## Test Slice Audit

- slice count: `39`
- positive slice count: `28`
- negative slice count: `2`
- easy hazard slice count: `10`
- core weak slices: `['horizon_100']`

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
| `domain_ETH_UCY_horizon_100` | `2052` | `0.00%` | `0.00%` | `2.38%` |
| `source_/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt_horizon_100` | `2052` | `0.00%` | `0.00%` | `2.38%` |
| `horizon_100` | `2406` | `0.00%` | `0.00%` | `2.15%` |

### Top Negative Slices

| slice | rows | delta vs graph | context rate | easy degradation |
| --- | ---: | ---: | ---: | ---: |
| `domain_ETH_UCY_horizon_50` | `2333` | `-0.02%` | `3.39%` | `0.00%` |
| `source_/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt_horizon_50` | `2333` | `-0.02%` | `3.39%` | `0.00%` |

## Interpretation

- Stage43-BV uses validation slice evidence to repair Stage43-BT context admissibility hazards.
- It does not retrain the BT MLP and does not tune on test.
- Future variant errors are validation/eval labels only, not inference inputs.
- This is a safety repair / diagnostic step; it is not a deployment update unless gates support it.
- Dataset-local/raw-frame 2.5D only; t100 remains raw-frame diagnostic; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `bt_precondition_passed` | `True` |
| `bu_precondition_passed` | `True` |
| `validation_slice_table_built` | `True` |
| `validation_only_repair_selected` | `True` |
| `test_eval_completed` | `True` |
| `bootstrap_completed` | `True` |
| `slice_audit_completed` | `True` |
| `t50_t100_reported` | `True` |
| `easy_safety_measured` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
