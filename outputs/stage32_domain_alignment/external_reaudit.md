# Stage32 External Reaudit

- source: `fresh_run`; Stage31 feature store is `cached_verified`.
- rows: `{'train': 119109, 'val': 7685, 'test': 3636}`
- coordinate unit: `dataset_local_coordinates`
- scale status: `unverified_weak_metric_diagnostic`
- horizon availability: `{'train': {10: 38978, 25: 33700, 50: 28743, 100: 17688}, 'val': {10: 2465, 25: 2175, 50: 1885, 100: 1160}, 'test': {10: 2020, 25: 1212, 50: 404}}`
- frame/step: `OpenTraj/TrajNet frame ids; raw horizon is frame-id delta or nearest future frame at/after requested delta.`
- agent type: `Pedestrian`
- scene/goal availability: `{'scene_packs': False, 'goal_candidates': False, 'goal_features_present_but_zero_filled': ['goal_count', 'nearest_goal_distance', 'goal_direction_alignment', 'goal_source_train_endpoint', 'goal_source_visual_prior', 'goal_directed_available']}`
- interaction features: `['density_visible_count', 'density_r20', 'density_r50', 'density_r100', 'nearest_neighbor_distance', 'mean_nearest3_distance', 'mean_nearest5_distance', 'min_ttc', 'max_closing_speed', 'closing_neighbor_count', 'nearest_goal_distance']`
- no leakage: `{'split_by_file': True, 'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'candidate_goals_used': False}`
