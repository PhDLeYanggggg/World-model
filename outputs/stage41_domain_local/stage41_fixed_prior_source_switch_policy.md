# Stage41 Fixed-Composer Prior Source Switch Policy

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- positive domains: `['ETH_UCY']`
- two-domain fixed-prior gate: `False`
- domains better than fixed composer on any core metric: `[]`
- two-domain beats-fixed gate: `False`

| domain | fixed policy | all ADE | t50 ADE | t100 ADE | hard ADE | easy | source switch | fixed delta all/t50/t100/hard | pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `ETH_UCY` | `gain_gate/bridge/old_shape` | 0.0163 | 0.0019 | 0.0043 | 0.0161 | 0.0000 | 0.005602 | -0.000075/0.000000/-0.000044/-0.000077 | `True` |
| `TrajNet` | `bridge/gain_gate/gain_gate` | 0.0381 | 0.0269 | 0.0138 | 0.0392 | 0.0000 | 0.000000 | 0.000000/0.000000/0.000000/0.000000 | `False` |

## Interpretation

- This experiment treats the fixed horizon composer as the safety prior and learns only residual source switches around that prior.
- It directly asks whether a learned source-switch model can improve the current deployable composer instead of merely staying positive versus the floor.
- Validation selects conservative thresholds; test is evaluated once.
- This remains protected dataset-local raw-frame 2.5D evidence, not Stage5C/SMC/metric/seconds/true-3D/foundation evidence.
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'val_selected_policy': True, 'stage5c_executed': False, 'smc_enabled': False}`
- claim boundary: `{'fixed_composer_prior_switch': True, 'learned_waypoint_shape_residual': True, 'protected_by_endpoint_bridge_or_floor': True, 'latent_generative_rollout': False, 'metric_or_seconds_claim': False, 'true_3d': False, 'foundation_world_model': False}`
