# Stage41 Pairwise Gain/Harm Shape Switch Policy

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- positive domains: `['ETH_UCY', 'TrajNet']`
- two-domain pairwise gate: `True`
- better than fixed composer on any core metric: `['TrajNet']`

| domain | all ADE | t50 ADE | t100 ADE | hard ADE | easy | shape gain all/t50/t100/hard | source distribution | sign acc | fixed delta all/t50/t100/hard | pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| `ETH_UCY` | 0.0161 | 0.0018 | 0.0043 | 0.0159 | 0.0000 | 0.000402/-0.000126/-0.000030/0.000413 | `{'bridge': 0.9974071673303083, 'old_shape': 4.630058338735068e-05, 'gain_gate': 0.0025465320863042873}` | 0.4262 | -0.000323/-0.000125/-0.000044/-0.000329 | `True` |
| `TrajNet` | 0.0382 | 0.0265 | 0.0142 | 0.0392 | 0.0000 | 0.000141/0.000000/0.000450/0.000155 | `{'bridge': 0.9994503984611157, 'old_shape': 0.0005496015388843089, 'gain_gate': 0.0}` | 0.5022 | 0.000023/-0.000456/0.000443/0.000025 | `True` |

## Interpretation

- This experiment is the direct repair attempt after absolute-ADE dynamic source ranking and simple calibration failed on ETH_UCY.
- It predicts pairwise gain and harm for switching from the protected bridge/Stage37 floor into each learned shape source.
- Validation selects conservative gain, harm, margin, and per-horizon switch-rate thresholds; test is evaluated once.
- If pairwise switching does not beat the fixed horizon composer, the fixed composer remains the safer deployable policy.
- This is still protected 2.5D dataset-local raw-frame evidence, not Stage5C/SMC/metric/seconds/true-3D/foundation evidence.
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'val_selected_policy': True, 'stage5c_executed': False, 'smc_enabled': False}`
- claim boundary: `{'pairwise_gain_harm_source_switch': True, 'learned_waypoint_shape_residual': True, 'protected_by_endpoint_bridge_or_floor': True, 'latent_generative_rollout': False, 'metric_or_seconds_claim': False, 'true_3d': False, 'foundation_world_model': False}`
