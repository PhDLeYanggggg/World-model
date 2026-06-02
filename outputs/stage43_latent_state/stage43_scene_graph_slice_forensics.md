# Stage43-BR Scene-Graph Slice Forensics

- source: `fresh_stage43_br_scene_graph_slice_forensics`
- result_source: `fresh_slice_forensics_from_stage43_bp_bq_checkpoints`
- verdict: `stage43_br_scene_graph_slice_forensics_pass_targeted_scene_signal`
- gate: `11 / 11`
- rows: `12000`
- scene over graph eligible slices: `8`
- scene over no_context eligible slices: `18`
- full over graph eligible slices: `5`
- best variant counts: `{'no_context': 2, 'scene_proxy_only': 7, 'graph_history_only': 17, 'scene_graph_full': 1}`
- deployable policy changed: `False`

## Slice Table

| slice | rows | best | no_context | scene | graph | full | scene-graph | full-graph |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all` | `12000` | `graph_history_only` | `32.73%` | `33.67%` | `36.91%` | `31.70%` | `-3.24%` | `-5.22%` |
| `source__Users_yangyue_Downloads_World_external_data_OpenTraj_datasets_TrajNet_Train_mot_PETS09-S2L1.txt` | `247` | `scene_proxy_only` | `-68.71%` | `-56.22%` | `-90.19%` | `-81.41%` | `33.97%` | `8.78%` |
| `domain_TrajNet_horizon_50` | `249` | `graph_history_only` | `12.08%` | `1.45%` | `17.44%` | `0.30%` | `-15.99%` | `-17.14%` |
| `domain_TrajNet_horizon_10` | `516` | `scene_proxy_only` | `9.87%` | `22.23%` | `6.27%` | `-1.18%` | `15.96%` | `-7.44%` |
| `domain_ETH_UCY_horizon_50` | `2333` | `graph_history_only` | `2.72%` | `1.06%` | `13.61%` | `2.60%` | `-12.55%` | `-11.01%` |
| `horizon_50` | `2888` | `graph_history_only` | `6.13%` | `3.63%` | `15.62%` | `4.32%` | `-11.99%` | `-11.30%` |
| `domain_ETH_UCY_horizon_25` | `2449` | `graph_history_only` | `47.63%` | `42.75%` | `50.99%` | `45.62%` | `-8.23%` | `-5.37%` |
| `domain_UCY_horizon_50` | `306` | `graph_history_only` | `19.83%` | `17.13%` | `24.50%` | `14.53%` | `-7.37%` | `-9.97%` |
| `horizon_25` | `3162` | `graph_history_only` | `45.13%` | `41.48%` | `48.53%` | `42.48%` | `-7.05%` | `-6.05%` |
| `domain_UCY_horizon_25` | `366` | `graph_history_only` | `48.59%` | `50.48%` | `57.40%` | `49.31%` | `-6.92%` | `-8.09%` |
| `domain_UCY_horizon_10` | `388` | `scene_graph_full` | `53.41%` | `62.87%` | `69.49%` | `74.14%` | `-6.62%` | `4.65%` |
| `domain_UCY` | `1271` | `graph_history_only` | `38.81%` | `41.86%` | `48.27%` | `44.48%` | `-6.41%` | `-3.79%` |
| `source__Users_yangyue_Downloads_World_external_data_OpenTraj_datasets_TrajNet_Train_crowds_crowds_zara03.txt` | `1271` | `graph_history_only` | `38.81%` | `41.86%` | `48.27%` | `44.48%` | `-6.41%` | `-3.79%` |
| `source__Users_yangyue_Downloads_World_external_data_OpenTraj_datasets_TrajNet_Train_biwi_biwi_hotel.txt` | `1008` | `graph_history_only` | `38.70%` | `36.51%` | `42.26%` | `28.86%` | `-5.75%` | `-13.41%` |
| `domain_ETH_UCY_horizon_10` | `2640` | `scene_proxy_only` | `62.61%` | `68.94%` | `64.48%` | `60.76%` | `4.47%` | `-3.71%` |
| `horizon_10` | `3544` | `scene_proxy_only` | `56.21%` | `63.57%` | `59.54%` | `56.63%` | `4.03%` | `-2.90%` |
| `easy` | `3607` | `graph_history_only` | `5.20%` | `5.22%` | `8.96%` | `-13.44%` | `-3.74%` | `-22.40%` |
| `domain_ETH_UCY` | `9474` | `graph_history_only` | `33.53%` | `34.10%` | `37.43%` | `32.38%` | `-3.34%` | `-5.06%` |
| `source__Users_yangyue_Downloads_World_external_data_OpenTraj_datasets_UCY_students03_obsmat.txt` | `9474` | `graph_history_only` | `33.53%` | `34.10%` | `37.43%` | `32.38%` | `-3.34%` | `-5.06%` |
| `not_easy` | `8393` | `graph_history_only` | `36.92%` | `38.00%` | `41.17%` | `38.57%` | `-3.17%` | `-2.60%` |
| `hard_failure` | `9371` | `graph_history_only` | `32.96%` | `34.85%` | `37.64%` | `34.15%` | `-2.79%` | `-3.49%` |
| `domain_TrajNet` | `1255` | `scene_proxy_only` | `16.48%` | `17.33%` | `14.86%` | `6.04%` | `2.47%` | `-8.82%` |
| `domain_ETH_UCY_horizon_100` | `2052` | `scene_proxy_only` | `-3.28%` | `-1.78%` | `-3.69%` | `-3.07%` | `1.91%` | `0.62%` |
| `horizon_100` | `2406` | `scene_proxy_only` | `-2.94%` | `-1.59%` | `-3.26%` | `-2.74%` | `1.67%` | `0.52%` |

## Interpretation

- This is row-level forensics over Stage43-BP checkpoints, not a new deployment policy.
- It identifies whether train-only scene proxies have slice-specific utility after BQ showed safe gated fusion still did not lift over graph-history.
- Future waypoints remain labels/eval only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `bp_precondition_passed` | `True` |
| `bq_precondition_passed` | `True` |
| `row_level_bp_checkpoint_replay_completed` | `True` |
| `slice_table_nonempty` | `True` |
| `eligible_slices_present` | `True` |
| `source_horizon_hard_easy_slices_present` | `True` |
| `scene_utility_measured` | `True` |
| `graph_history_best_count_measured` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
