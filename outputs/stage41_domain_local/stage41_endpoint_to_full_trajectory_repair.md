# Stage41 Endpoint-To-Full-Trajectory Repair

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- positive domains: `['ETH_UCY', 'TrajNet']`
- two-domain endpoint-to-full gate: `True`

| domain | variant | all ADE | t50 ADE | t100 ADE | hard ADE | easy | multi all | collision d005 | pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ETH_UCY` | `all_horizons` | 0.0157 | 0.0019 | 0.0043 | 0.0155 | 0.0000 | 0.0159 | -0.0015 | `True` |
| `TrajNet` | `all_horizons` | 0.0380 | 0.0265 | 0.0138 | 0.0391 | 0.0000 | 0.0404 | -0.0029 | `True` |

## Interpretation

- This repair checks whether domain-local endpoint neural dynamics can bridge the ETH_UCY t50 full-waypoint failure when scored against reconstructed actual waypoint labels.
- It does not claim learned full-waypoint shape dynamics; the rollout between current point and predicted endpoint is linear.
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'val_selected_policy': True, 'stage5c_executed': False, 'smc_enabled': False}`
- claim boundary: `{'endpoint_neural_dynamics': True, 'linear_waypoint_bridge': True, 'learned_full_waypoint_shape': False, 'latent_generative_rollout': False, 'metric_or_seconds_claim': False, 'true_3d': False, 'foundation_world_model': False}`
