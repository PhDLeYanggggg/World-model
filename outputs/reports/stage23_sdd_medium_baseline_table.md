# Stage 23 SDD Medium Baseline Table

- mode: `quick-plus`
- t+100 remains raw-frame pixel-space; effective seconds unknown.
- selector oracle headroom vs damped velocity: `{'cross_scene': 0.15804028771313627, 'within_scene': 0.41022737821256927}`

| split_type | horizon | strongest | FDE |
| --- | --- | --- | --- |
| cross_scene | 10 | damped_velocity | 4.6007 |
| cross_scene | 25 | damped_velocity | 15.1996 |
| cross_scene | 50 | damped_velocity | 28.0318 |
| cross_scene | 100 | damped_velocity | 61.1741 |
| within_scene | 10 | damped_velocity | 4.4406 |
| within_scene | 25 | damped_velocity | 14.4055 |
| within_scene | 50 | scene_clamped_baseline | 21.2937 |
| within_scene | 100 | scene_clamped_baseline | 45.1828 |
