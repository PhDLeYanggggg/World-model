# Stage41 Learned Waypoint-Shape Bridge

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- positive domains: `['ETH_UCY', 'TrajNet']`
- two-domain learned-shape gate: `True`

| domain | shape mode | all ADE | t50 ADE | t100 ADE | hard ADE | easy | shape gain all/t50/t100/hard | shape switch | collision d005 | pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `ETH_UCY` | `all_waypoints` | 0.0157 | 0.0019 | 0.0043 | 0.0155 | 0.0000 | 0.000005/0.000000/0.000014/0.000005 | 0.000046 | -0.0015 | `True` |
| `TrajNet` | `intermediate_only` | 0.0382 | 0.0265 | 0.0145 | 0.0393 | 0.0000 | 0.000226/0.000000/0.000718/0.000248 | 0.001099 | -0.0029 | `True` |

## Interpretation

- This experiment is the next step after the endpoint-to-full linear bridge: it learns a waypoint-shape residual around the endpoint neural bridge from past-only features.
- The model is still protected by the endpoint bridge/floor fallback; future waypoints are labels/eval only.
- A positive learned-shape gain versus the linear bridge is required before claiming learned waypoint-shape contribution.
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'val_selected_policy': True, 'stage5c_executed': False, 'smc_enabled': False}`
- claim boundary: `{'endpoint_neural_dynamics': True, 'learned_waypoint_shape_residual': True, 'stage37_or_floor_protected': True, 'latent_generative_rollout': False, 'metric_or_seconds_claim': False, 'true_3d': False, 'foundation_world_model': False}`
