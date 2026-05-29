# Stage43-B Latent-State Dataset Contract

- source: `fresh_stage43_b_latent_state_dataset_contract`
- verdict: `stage43_b_latent_state_dataset_contract_pass`
- gate: `12 / 12`
- endpoint latent-state training ready: `True`
- full-waypoint supervised training ready: `False`

## Split Summary

| split | role | rows | domains | horizons | K8 history | hard | failure | easy | alignment |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| train | `supervised_training` | 158942 | `{'ETH_UCY': 150798, 'TrajNet': 8144}` | `{'10': 46604, '25': 42907, '50': 38943, '100': 30488}` | 128439 | 131433 | 64528 | 39736 | True |
| val | `supervised_training` | 112746 | `{'TrajNet': 112746}` | `{'10': 37683, '25': 32059, '50': 26756, '100': 16248}` | 67879 | 78283 | 39068 | 35967 | True |
| test | `official_eval_dataset_local_raw_frame` | 66303 | `{'UCY': 66303}` | `{'10': 21267, '25': 18765, '50': 16263, '100': 10008}` | 41283 | 45917 | 22891 | 20798 | True |

## Token Groups

- `context_observation`: dataset, scene_id, source_file, agent_id, frame_id, current_x, current_y, horizon, scale, split, data_role
- `agent_history`: history_x, history_y, history_dx, history_dy, history_speed, history_accel, history_heading, history_valid_mask, history_curvature, history_turn_angle, history_stop_go, history_dwell, history_path_length, history_velocity_decay
- `all_agent_graph`: history_neighbor_count, history_min_neighbor_dist, history_density, history_TTC, history_closing_speed
- `scene_goal_proxy`: prototype_vectors, prototype_likelihood, prototype_entropy, goal_ambiguity, prototype_distance, prototype_angle
- `baseline_rollout_family`: baseline_family_prediction, baseline_family_y_fde, baseline_family_relative_y, strongest_idx, oracle_idx, stage37_selected_family, stage35_selected_family
- `safety_floor_state`: easy, hard, failure, oracle_margin, stage37_confidence, stage36_predicted_gain, stage36_hard_prob, stage36_fail_prob, stage36_easy_prob
- `labels_only`: future_endpoint_x, future_endpoint_y, future_relative_y, future_fde_by_baseline, full_waypoint_xy_partial, waypoint_valid_partial, occupancy_density_proxy, failure_label, gain_label, harm_label

## Label Boundary

- Future endpoint, future relative error, full-waypoint, occupancy/density, failure/gain/harm are label/loss/eval targets only.
- They are not listed in any inference input token group.
- Full-waypoint status: `partial_eval_cache`.
- Full-waypoint limitation: Full-waypoint labels are present in the Stage42 current source-level evaluation cache, but a train/val full-waypoint latent-state supervised cache is not yet frozen.

## Gate

| gate | passed |
| --- | --- |
| stage43_a_precondition_passed | True |
| all_split_artifacts_exist | True |
| all_split_rows_align | True |
| train_val_test_roles_explicit | True |
| history_windows_available | True |
| goal_prototypes_available | True |
| baseline_family_available | True |
| endpoint_labels_available_all_splits | True |
| full_waypoint_limitation_recorded | True |
| labels_separated_from_inputs | True |
| no_future_or_test_leakage | True |
| no_metric_seconds_stage5c_smc_claim | True |

## Decision

Endpoint/failure/gain/harm/occupancy latent-state dataset contract is ready; full-waypoint supervised training remains blocked until train/val full-waypoint labels are frozen.

No Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.
