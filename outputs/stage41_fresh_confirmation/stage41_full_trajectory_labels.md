# Stage41 Full-Trajectory Label Reconstruction

- source: `fresh_run`
- future waypoints are label/evaluation only; inputs remain past-only all-agent tokens.
- waypoint fractions: `[0.25, 0.5, 0.75, 1.0]`
- splits: `{'train': {'rows': 219667, 'full_waypoint_rows': 151784, 'endpoint_only_rows': 67883, 'missing_track_rows': 0, 'interaction_positive': 157544, 'occupancy_positive': 101331, 'physical_valid': 219666, 'domains': {'ETH_UCY': 108794, 'TrajNet': 63650, 'UCY': 47223}, 't50': 52832, 't100': 36540}, 'val': {'rows': 53256, 'full_waypoint_rows': 36759, 'endpoint_only_rows': 16497, 'missing_track_rows': 0, 'interaction_positive': 37154, 'occupancy_positive': 21793, 'physical_valid': 53256, 'domains': {'ETH_UCY': 16103, 'TrajNet': 37153}, 't50': 13101, 't100': 8859}, 'test': {'rows': 55528, 'full_waypoint_rows': 38802, 'endpoint_only_rows': 16726, 'missing_track_rows': 0, 'interaction_positive': 38208, 'occupancy_positive': 25035, 'physical_valid': 55528, 'domains': {'ETH_UCY': 25901, 'TrajNet': 20087, 'UCY': 9540}, 't50': 13689, 't100': 9905}}`
- no leakage: `{'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False}`
