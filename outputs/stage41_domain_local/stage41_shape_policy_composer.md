# Stage41 Domain/Horizon Shape-Policy Composer

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- positive domains: `['ETH_UCY', 'TrajNet']`
- two-domain composer gate: `True`

| domain | selected short/t50/t100 | all ADE | t50 ADE | t100 ADE | hard ADE | easy | shape gain all/t50/t100/hard | shape switch | pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `ETH_UCY` | `gain_gate/bridge/old_shape` | 0.0164 | 0.0019 | 0.0043 | 0.0162 | 0.0000 | 0.000730/0.000000/0.000014/0.000747 | 0.005649 | `True` |
| `TrajNet` | `bridge/gain_gate/gain_gate` | 0.0381 | 0.0269 | 0.0138 | 0.0392 | 0.0000 | 0.000117/0.000469/0.000000/0.000129 | 0.000824 | `True` |

## Interpretation

- The composer addresses the previous mixed result by selecting the best source per horizon family on validation only.
- It can choose the pure bridge, the residual-norm learned-shape policy, or the train-fitted gain/harm shape policy for short, t50, and t100 rows separately.
- This is still a protected full-waypoint bridge, not an unprotected latent generative rollout and not a metric/seconds/true-3D/foundation result.
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'val_selected_policy': True, 'stage5c_executed': False, 'smc_enabled': False}`
- claim boundary: `{'domain_horizon_composer': True, 'learned_waypoint_shape_residual': True, 'protected_by_endpoint_bridge_or_floor': True, 'latent_generative_rollout': False, 'metric_or_seconds_claim': False, 'true_3d': False, 'foundation_world_model': False}`
