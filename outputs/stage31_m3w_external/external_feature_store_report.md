# Stage31 External Feature Store Report

- source: `fresh_run` conversion; raw local files and Stage26 schema are `cached_verified` inputs.
- Stage5C executed: `False`; SMC enabled: `False`.
- dataset: `OpenTraj TrajNet non-SDD pedestrian subsets`
- rows: `{'train': 119109, 'val': 7685, 'test': 3636}`
- horizon counts: `{'train': {10: 38978, 25: 33700, 50: 28743, 100: 17688}, 'val': {10: 2465, 25: 2175, 50: 1885, 100: 1160}, 'test': {10: 2020, 25: 1212, 50: 404}}`
- coordinate unit: `dataset_local_coordinates`
- metric status: `unverified_weak_metric_diagnostic`
- strongest baseline readiness: `constant_velocity_causal_fd`
- no leakage: `{'split_by_file': True, 'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'candidate_goals_used': False}`
- ready: `True`
