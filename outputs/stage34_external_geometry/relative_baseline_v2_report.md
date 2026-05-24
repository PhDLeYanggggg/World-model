# Stage34 Relative Baseline v2 Report

- source: `fresh_run`

| normalizer | split | strongest | rows |
| --- | --- | --- | ---: |
| history_path_length | train | damped_velocity | 119109 |
| history_path_length | val | scene_clamped_baseline | 7685 |
| history_path_length | test | scene_clamped_baseline | 3636 |
| speed_horizon | train | damped_velocity | 119109 |
| speed_horizon | val | scene_clamped_baseline | 7685 |
| speed_horizon | test | scene_clamped_baseline | 3636 |
| scene_scale | train | damped_velocity | 119109 |
| scene_scale | val | damped_velocity | 7685 |
| scene_scale | test | constant_position | 3636 |
| median_train_displacement | train | damped_velocity | 119109 |
| median_train_displacement | val | damped_velocity | 7685 |
| median_train_displacement | test | constant_position | 3636 |

- selected normalizer: `scene_scale`
