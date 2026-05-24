# Stage32 Domain Normalization Report

- source: `fresh_run`
- normalizations: `['raw_dataset_local', 'per_scene_zscore', 'velocity_scale', 'path_length_speed', 'robust_quantile']`
- aligned feature families: `['x/y proxies', 'vx/vy/speed', 'accel', 'heading/curvature', 'density/nearest/TTC', 'horizon', 'agent_type']`
- No metric/seconds claim: coordinates remain dataset-local or SDD pixel raw-frame.
