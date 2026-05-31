# Stage43-Z Latent Token Schema Coverage

- source: `fresh_stage43_z_latent_token_schema_coverage`
- result_source: `fresh_schema_audit_from_cached_verified_stage43_full_waypoint_cache`
- gate: `12 / 12`
- verdict: `stage43_z_latent_token_schema_coverage_pass`
- protected schema supported: `True`
- standalone ungated deployment: `False`

## Feature Schema

- feature dim: `162`
- feature schema hash: `fba36ccddae43a4776793fb92ef305162abf4649f632f0d9696a463bac31022b`
- train rows sampled for schema: `512`

## Split Hashes

| split | rows | row hash | schema hash | domains | horizons |
| --- | ---: | --- | --- | --- | --- |
| train | 146809 | `92421511e9e2` | `65e5ad4c1742` | `{'ETH_UCY': 63602, 'TrajNet': 73667, 'UCY': 9540}` | `{'10': 46716, '25': 41248, '50': 35467, '100': 23378}` |
| val | 101446 | `788e001810b7` | `9ae5728c714a` | `{'ETH_UCY': 16611, 'TrajNet': 37612, 'UCY': 47223}` | `{'10': 32706, '25': 28703, '50': 24741, '100': 15296}` |
| test | 89736 | `1613ed7188f2` | `bc22ada03985` | `{'ETH_UCY': 70585, 'TrajNet': 9611, 'UCY': 9540}` | `{'10': 26132, '25': 23780, '50': 21754, '100': 18070}` |

## Token Coverage

| token group | kind | status | claim boundary |
| --- | --- | --- | --- |
| `agent_state` | `inference_input` | `covered` | covered |
| `agent_history` | `inference_input` | `covered` | covered |
| `all_agent_current_state` | `row_metadata_grouping` | `covered` | partial_grouped_rows_not_explicit_tensor |
| `neighbor_graph` | `inference_proxy` | `covered` | proxy_only_not_full_graph_tensor |
| `scene_patch` | `missing_modality` | `recorded_gap` | missing_explicit_scene_image_or_raster_token |
| `scene_sdf` | `missing_modality` | `recorded_gap` | missing_explicit_scene_sdf_token |
| `goal_region` | `inference_proxy` | `covered` | scene_agnostic_goal_proxy_covered |
| `interaction_edge` | `inference_proxy_and_label_proxy` | `covered` | proxy_only_not_human_interaction_annotation |
| `baseline_rollout` | `inference_input` | `covered` | baseline_family_endpoint_rollouts_covered |
| `safety_floor_prediction` | `inference_input` | `covered` | stage37_stage42_floor_endpoint_covered |
| `domain_source_horizon` | `inference_input_and_metadata` | `covered` | domain_horizon_features_source_metadata_covered |
| `time_frame` | `metadata_only` | `covered` | frame_metadata_covered_not_seconds_verified |
| `future_endpoint_label` | `label_only` | `label_only_separated` | label_only_not_input |
| `future_full_waypoint_label` | `label_only` | `label_only_separated` | label_only_not_input |
| `occupancy_density_label` | `proxy_label` | `covered` | causal_history_density_proxy_not_future_occupancy |
| `failure_gain_harm_label` | `proxy_label` | `covered` | covered_training_labels |
| `mask_validity` | `label_and_input_mask` | `covered` | covered |

## Explicit Gaps

- `explicit_scene_image_raster_token`: `missing`
- `explicit_scene_sdf_token`: `missing`
- `full_all_agent_graph_tensor`: `missing_proxy_only`
- `future_occupancy_true_label`: `missing_proxy_only`
- `true_physical_validity_label`: `missing_proxy_only`
- `human_interaction_annotation`: `missing_proxy_only`
- `verified_metric_or_seconds_calibration`: `missing`

## No-Leakage Boundary

- Future endpoints and full waypoints are labels/eval targets only.
- Scene raster/image/SDF tokens are not present in the current Stage43 full-waypoint cache.
- Neighbor and interaction context are causal proxy features, not a full graph tensor or human interaction annotation.
- Density is a causal history-density proxy, not future occupancy.
- Smoothness/validity remains diagnostic proxy evidence, not true physical validity.

## Gate

| gate | passed |
| --- | --- |
| stage43_y_head_suite_passed | True |
| split_caches_exist | True |
| row_hashes_recorded | True |
| feature_schema_recorded | True |
| core_causal_inputs_covered | True |
| all_agent_and_neighbor_scope_honest | True |
| future_labels_separated_from_inputs | True |
| scene_raster_sdf_gaps_recorded | True |
| proxy_boundaries_recorded | True |
| no_future_or_test_leakage | True |
| claim_boundary_preserved | True |
| protected_not_standalone | True |

No Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.
