# Stage34 External Row Geometry Report

- source: `fresh_run`; Stage31 feature rows and raw OpenTraj files are `cached_verified`.
- future endpoint is stored only as supervision/evaluation label, not an inference feature.
- row geometry complete: `True`
- splits: `{'train': {'expected_feature_rows': 119109, 'geometry_rows': 119109, 'aligned_to_feature_store': True, 'scenes': 2, 'agents': 892, 'horizon_counts': {10: 38978, 25: 33700, 50: 28743, 100: 17688}, 'has_future_endpoint_label': True, 'future_endpoint_used_as_inference_feature': False}, 'val': {'expected_feature_rows': 7685, 'geometry_rows': 7685, 'aligned_to_feature_store': True, 'scenes': 1, 'agents': 145, 'horizon_counts': {10: 2465, 25: 2175, 50: 1885, 100: 1160}, 'has_future_endpoint_label': True, 'future_endpoint_used_as_inference_feature': False}, 'test': {'expected_feature_rows': 3636, 'geometry_rows': 3636, 'aligned_to_feature_store': True, 'scenes': 2, 'agents': 220, 'horizon_counts': {10: 2020, 25: 1212, 50: 404}, 'has_future_endpoint_label': True, 'future_endpoint_used_as_inference_feature': False}}`
- no leakage: `{'future_endpoint_in_features': False, 'future_endpoint_label_only': True, 'central_velocity': False, 'test_endpoint_goals': False}`
