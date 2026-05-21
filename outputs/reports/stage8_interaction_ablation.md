# Stage 8 Interaction v2 Ablation

| variant | samples | future_nn_mae | close_pass_F1 |
| --- | --- | --- | --- |
| no_interaction | 12 | 4.068562 | 0.0 |
| scalar_interaction | 12 | 0.916598 | 0.0 |
| graph_interaction | 12 | 0.940188 | 0.0 |
| graph_interaction_scene_goal | 12 | 0.940188 | 0.0 |


Interaction trajectory lift is determined in `metrics_stage8.json`, not from auxiliary metrics alone.
