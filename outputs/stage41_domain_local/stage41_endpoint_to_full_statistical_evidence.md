# Stage41 Endpoint-To-Full Statistical Evidence

- source: `fresh_run`
- bootstrap_n: `2000`
- positive domains: `['ETH_UCY', 'TrajNet']`
- two-domain statistical gate: `True`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`

| domain | all ADE | all low | t50 ADE | t50 low | t100 ADE | t100 low | hard ADE | hard low | multi low | FDE all low | FDE t50 low | easy | pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ETH_UCY` | 0.0157 | 0.0150 | 0.0019 | 0.0014 | 0.0043 | 0.0032 | 0.0155 | 0.0148 | 0.0151 | 0.0154 | 0.0020 | 0.0000 | `True` |
| `TrajNet` | 0.0380 | 0.0338 | 0.0265 | 0.0186 | 0.0138 | 0.0077 | 0.0391 | 0.0344 | 0.0359 | 0.0339 | 0.0258 | 0.0000 | `True` |

## Interpretation

- This strengthens the endpoint-to-full bridge by adding per-domain bootstrap lower bounds for actual waypoint ADE/FDE.
- It does not convert the claim into learned full-waypoint shape dynamics; the waypoint path remains a protected linear bridge from endpoint neural dynamics.
- Raw ungated full-row neural safety is still not claimed.
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'val_selected_policy': True}`
- claim boundary: `{'endpoint_neural_dynamics': True, 'linear_waypoint_bridge': True, 'learned_full_waypoint_shape': False, 'ungated_full_row_neural_safety': False, 'latent_generative_rollout': False, 'metric_or_seconds_claim': False, 'true_3d': False, 'foundation_world_model': False, 'stage5c_executed': False, 'smc_enabled': False}`
