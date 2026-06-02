# Stage43-BW Context Hazard Attribution Guard

- source: `fresh_stage43_bw_context_hazard_attribution_guard`
- result_source: `fresh_validation_selected_context_hazard_attribution_guard`
- verdict: `stage43_bw_context_hazard_attribution_pass_floor_inherent_risk`
- gate: `13 / 13`
- selected guard: `guard_domain_horizon_rate_0.20_plus_block_t100`
- safe validation candidates: `25 / 43`
- source overlap: `{'val_source_count': 4, 'test_source_count': 4, 'overlap_count': 0, 'overlap_examples': [], 'held_out_source_level': True}`
- deployable policy changed: `False`

## Test Metrics

- all full-waypoint ADE improvement: `38.78%`
- t50 full-waypoint ADE improvement: `15.60%`
- t100 raw-frame diagnostic improvement: `-3.26%`
- hard/failure improvement: `39.41%`
- easy degradation: `0.00%`
- switch rate: `57.48%`
- context counts on test: `{'graph_history_only': 10522, 'scene_graph_full': 512, 'scene_proxy_only': 966}`
- blocked context rows on test: `200`

## Delta Vs Graph-History-Only

- all delta: `1.86%`
- t50 delta: `-0.02%`
- t100 raw-frame diagnostic delta: `0.00%`
- hard/failure delta: `1.77%`
- easy degradation delta: `0.00%`

## Absolute Easy Hazard Attribution

- graph-history absolute easy hazard slices: `11`
- BT unrepaired absolute easy hazard slices: `7`
- selected guard absolute easy hazard slices: `10`

## Context-Induced Easy Hazard

- BT unrepaired context-induced hazard slices: `12`
- selected guard context-induced hazard slices: `9`

### Top Selected-Guard Context Harm Slices

| slice | rows | easy rows | context easy rows | context harm rate | mean context harm | mean easy delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `source_horizon_source_horizon:/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/mot/PETS09-S2L1.txt|h:10` | `178` | `45` | `34` | `41.18%` | `0.036060` | `-0.005197` |
| `source_source:/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet/Train/mot/PETS09-S2L1.txt` | `247` | `48` | `35` | `40.00%` | `0.035030` | `-0.007063` |
| `domain_horizon_domain_horizon:TrajNet|h:10` | `516` | `174` | `58` | `29.31%` | `0.022712` | `-0.009313` |
| `domain_domain:TrajNet` | `1255` | `461` | `62` | `27.42%` | `0.021247` | `-0.004045` |
| `horizon_horizon:10` | `3544` | `945` | `160` | `21.88%` | `0.013887` | `-0.004756` |
| `domain_horizon_domain_horizon:ETH_UCY|h:10` | `2640` | `697` | `86` | `16.28%` | `0.009531` | `-0.003187` |
| `source_horizon_source_horizon:/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt|h:10` | `2640` | `697` | `86` | `16.28%` | `0.009531` | `-0.003187` |
| `domain_domain:ETH_UCY` | `9474` | `2888` | `126` | `15.87%` | `0.009095` | `-0.001230` |
| `source_source:/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt` | `9474` | `2888` | `126` | `15.87%` | `0.009095` | `-0.001230` |
| `domain_horizon_domain_horizon:ETH_UCY|h:25` | `2449` | `818` | `29` | `13.79%` | `0.008577` | `-0.001296` |
| `source_horizon_source_horizon:/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/UCY/students03/obsmat.txt|h:25` | `2449` | `818` | `29` | `13.79%` | `0.008577` | `-0.001296` |
| `horizon_horizon:25` | `3162` | `1017` | `33` | `12.12%` | `0.007538` | `-0.001283` |

## Validation Candidate Guards

| candidate | safe | score | all delta | t50 delta | t100 delta | hard delta | easy delta | context hazards | blocked rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `guard_domain_horizon_rate_0.20_plus_block_t100` | `True` | `0.06395` | `3.40%` | `0.27%` | `0.00%` | `2.71%` | `0.00%` | `0` | `373` |
| `guard_domain_horizon_rate_0.30_plus_block_t100` | `True` | `0.06395` | `3.40%` | `0.27%` | `0.00%` | `2.71%` | `0.00%` | `0` | `373` |
| `guard_source_horizon_rate_0.20_plus_block_t100` | `True` | `0.06395` | `3.40%` | `0.27%` | `0.00%` | `2.71%` | `0.00%` | `0` | `373` |
| `guard_source_horizon_rate_0.30_plus_block_t100` | `True` | `0.06395` | `3.40%` | `0.27%` | `0.00%` | `2.71%` | `0.00%` | `0` | `373` |
| `guard_domain_horizon_rate_0.20` | `True` | `0.06388` | `3.40%` | `0.27%` | `-0.01%` | `2.70%` | `0.00%` | `0` | `371` |
| `guard_domain_horizon_rate_0.30` | `True` | `0.06388` | `3.40%` | `0.27%` | `-0.01%` | `2.70%` | `0.00%` | `0` | `371` |
| `guard_source_horizon_rate_0.20` | `True` | `0.06388` | `3.40%` | `0.27%` | `-0.01%` | `2.70%` | `0.00%` | `0` | `371` |
| `guard_source_horizon_rate_0.30` | `True` | `0.06388` | `3.40%` | `0.27%` | `-0.01%` | `2.70%` | `0.00%` | `0` | `371` |
| `guard_horizon_rate_0.20_plus_block_t100` | `True` | `0.05414` | `3.03%` | `0.00%` | `0.00%` | `2.46%` | `0.00%` | `0` | `778` |
| `guard_horizon_rate_0.20` | `True` | `0.05407` | `3.03%` | `0.00%` | `-0.01%` | `2.46%` | `0.00%` | `0` | `776` |
| `guard_domain_rate_0.20` | `True` | `0.04538` | `2.33%` | `0.27%` | `0.00%` | `1.99%` | `0.00%` | `0` | `982` |
| `guard_source_rate_0.20` | `True` | `0.04538` | `2.33%` | `0.27%` | `0.00%` | `1.99%` | `0.00%` | `0` | `982` |
| `guard_domain_rate_0.20_plus_block_t100` | `True` | `0.04538` | `2.33%` | `0.27%` | `0.00%` | `1.99%` | `0.00%` | `0` | `983` |
| `guard_source_rate_0.20_plus_block_t100` | `True` | `0.04538` | `2.33%` | `0.27%` | `0.00%` | `1.99%` | `0.00%` | `0` | `983` |
| `guard_domain_horizon_rate_0.40_plus_block_t100` | `False` | `0.04142` | `3.56%` | `0.27%` | `0.00%` | `2.76%` | `0.00%` | `5` | `44` |
| `guard_source_horizon_rate_0.40_plus_block_t100` | `False` | `0.04142` | `3.56%` | `0.27%` | `0.00%` | `2.76%` | `0.00%` | `5` | `44` |
| `guard_domain_horizon_rate_0.40` | `False` | `0.04135` | `3.56%` | `0.27%` | `-0.01%` | `2.76%` | `0.00%` | `5` | `42` |
| `guard_source_horizon_rate_0.40` | `False` | `0.04135` | `3.56%` | `0.27%` | `-0.01%` | `2.76%` | `0.00%` | `5` | `42` |
| `guard_horizon_rate_0.30_plus_block_t100` | `False` | `0.03687` | `3.51%` | `0.00%` | `0.00%` | `2.69%` | `0.00%` | `5` | `94` |
| `guard_horizon_rate_0.30` | `False` | `0.03679` | `3.51%` | `0.00%` | `-0.01%` | `2.68%` | `0.00%` | `5` | `92` |

## Bootstrap Delta Vs Graph-History-Only

- bootstrap n: `2000`

| metric | rows | low | mean | high |
| --- | ---: | ---: | ---: | ---: |
| `all_delta_vs_graph` | `12000` | `1.65%` | `1.86%` | `2.08%` |
| `t50_delta_vs_graph` | `2888` | `-0.33%` | `-0.01%` | `0.26%` |
| `t100_raw_frame_delta_vs_graph` | `2406` | `0.00%` | `0.00%` | `0.00%` |
| `hard_failure_delta_vs_graph` | `9371` | `1.56%` | `1.77%` | `1.99%` |
| `easy_degradation_delta_vs_graph` | `3607` | `0.00%` | `0.00%` | `0.00%` |

## Interpretation

- Stage43-BW separates floor-inherent absolute easy risk from context-induced harm.
- Source-level hazard keys are reported with held-out source overlap so source-key guards are not mistaken for transferable safety.
- This is not a deployment update; it is a safety attribution step for the protected multimodal latent-state track.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `bt_precondition_present` | `True` |
| `bv_precondition_present` | `True` |
| `validation_only_guard_selected` | `True` |
| `source_overlap_reported` | `True` |
| `absolute_floor_hazard_attributed` | `True` |
| `context_induced_hazard_measured` | `True` |
| `context_hazard_not_worse_than_bt` | `True` |
| `test_eval_completed` | `True` |
| `bootstrap_completed` | `True` |
| `easy_safety_measured` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
