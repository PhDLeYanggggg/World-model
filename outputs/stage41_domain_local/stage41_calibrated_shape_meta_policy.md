# Stage41 Calibrated Shape Source Meta-Policy

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- positive domains: `['ETH_UCY', 'TrajNet']`
- two-domain calibrated meta gate: `True`

| domain | selected calibration | all ADE | t50 ADE | t100 ADE | hard ADE | easy | shape gain all/t50/t100/hard | rank acc | fixed delta all/t50/t100/hard | pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| `ETH_UCY` | `none` | 0.0163 | 0.0018 | 0.0043 | 0.0161 | 0.0000 | 0.000629/-0.000126/-0.000030/0.000650 | 0.0193 | -0.000099/-0.000125/-0.000044/-0.000095 | `True` |
| `TrajNet` | `none` | 0.0383 | 0.0267 | 0.0145 | 0.0394 | 0.0000 | 0.000287/0.000246/0.000718/0.000316 | 0.9997 | 0.000164/-0.000217/0.000708/0.000180 | `True` |

## Interpretation

- This experiment explicitly targets the ETH_UCY ranking collapse from the uncalibrated dynamic meta-policy.
- Calibration modes are selected on validation only; test is evaluated once.
- If calibrated ranking improves one domain but harms another, the fixed composer remains the safer deployable default.
- This remains protected 2.5D evidence, not Stage5C/SMC/metric/seconds/true-3D/foundation evidence.
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'val_selected_policy': True, 'stage5c_executed': False, 'smc_enabled': False}`
- claim boundary: `{'calibrated_dynamic_source_meta_policy': True, 'learned_waypoint_shape_residual': True, 'protected_by_endpoint_bridge_or_floor': True, 'latent_generative_rollout': False, 'metric_or_seconds_claim': False, 'true_3d': False, 'foundation_world_model': False}`
