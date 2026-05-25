# Stage41 Dynamic Shape Source Meta-Policy

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- positive domains: `['ETH_UCY', 'TrajNet']`
- two-domain dynamic meta gate: `True`

| domain | all ADE | t50 ADE | t100 ADE | hard ADE | easy | shape gain all/t50/t100/hard | source distribution | rank acc | pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| `ETH_UCY` | 0.0163 | 0.0018 | 0.0043 | 0.0161 | 0.0000 | 0.000629/-0.000126/-0.000030/0.000650 | `{'bridge': 0.9871284378183165, 'old_shape': 4.630058338735068e-05, 'gain_gate': 0.01282526159829614}` | 0.0193 | `True` |
| `TrajNet` | 0.0383 | 0.0267 | 0.0145 | 0.0394 | 0.0000 | 0.000287/0.000246/0.000718/0.000316 | `{'bridge': 0.9986259961527892, 'old_shape': 0.0010992030777686177, 'gain_gate': 0.0002748007694421544}` | 0.9997 | `True` |

## Interpretation

- This experiment trains a per-row source cost model over bridge, previous learned-shape, and gain-gate shape sources.
- The model is trained with train future-waypoint labels only; validation chooses gain/margin/source-rate thresholds; test is evaluated once.
- This is a stronger dynamic policy than the fixed horizon composer if it preserves easy cases while allowing source switches with positive shape gain.
- It is still protected 2.5D world-state evidence, not Stage5C, SMC, metric prediction, seconds-level prediction, true 3D, or foundation evidence.
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'val_selected_policy': True, 'stage5c_executed': False, 'smc_enabled': False}`
- claim boundary: `{'dynamic_source_meta_policy': True, 'learned_waypoint_shape_residual': True, 'protected_by_endpoint_bridge_or_floor': True, 'latent_generative_rollout': False, 'metric_or_seconds_claim': False, 'true_3d': False, 'foundation_world_model': False}`
