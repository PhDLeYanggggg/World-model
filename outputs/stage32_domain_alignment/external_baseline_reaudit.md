# Stage32 External Baseline Reaudit

- source: `fresh_run`
- Normalized FDE is diagnostic; original external evaluation remains dataset-local.

| normalization | test strongest | test rows |
| --- | --- | ---: |
| raw_dataset_local | constant_velocity_causal_fd | 3636 |
| per_scene_zscore | constant_velocity_causal_fd | 3636 |
| velocity_scale | constant_velocity_causal_fd | 3636 |
| path_length_speed | constant_velocity_causal_fd | 3636 |
| robust_quantile | constant_velocity_causal_fd | 3636 |
