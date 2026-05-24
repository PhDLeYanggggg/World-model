# Stage33 Coordinate-Invariant Feature Report

- source: `fresh_run`; Stage26/Stage31 feature stores are `cached_verified`.
- schema hash: `8e24edb1e76194a3fe72ecca85b0fad7970a7d68653a20b337a79f3d2d27323c`
- rows: `{'sdd': {'train': {'rows': 40000, 'features': 37, 'finite_fraction': 1.0}, 'val': {'rows': 20000, 'features': 37, 'finite_fraction': 1.0}, 'test': {'rows': 100000, 'features': 37, 'finite_fraction': 1.0}}, 'external': {'train': {'rows': 119109, 'features': 37, 'finite_fraction': 1.0}, 'val': {'rows': 7685, 'features': 37, 'finite_fraction': 1.0}, 'test': {'rows': 3636, 'features': 37, 'finite_fraction': 1.0}}}`
- feature count: `37`
- forbidden inputs: `['future_endpoint', 'future_goal_label', 'central_velocity', 'test_endpoint_goals', 'ground_truth_future']`
- Coordinates remain pixel/dataset-local; no metric or seconds claim.
