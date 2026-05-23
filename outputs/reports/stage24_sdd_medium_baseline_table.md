# Stage 24 SDD Medium Baseline Table

- mode: `medium`
- SDD remains pixel-space; t+100 remains raw-frame horizon.
- strongest baseline still damped velocity somewhere: `True`
- within_scene strongest scene_clamped somewhere: `True`

| split_type | horizon | strongest | FDE |
| --- | --- | --- | ---: |
| cross_scene | 10 | damped_velocity | 5.5966 |
| cross_scene | 25 | damped_velocity | 13.2470 |
| cross_scene | 50 | damped_velocity | 27.4226 |
| cross_scene | 100 | damped_velocity | 59.9263 |
| within_scene | 10 | damped_velocity | 4.9716 |
| within_scene | 25 | damped_velocity | 12.7201 |
| within_scene | 50 | damped_velocity | 29.4528 |
| within_scene | 100 | scene_clamped_baseline | 62.5456 |
