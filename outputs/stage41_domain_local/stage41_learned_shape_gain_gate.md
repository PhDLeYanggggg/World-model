# Stage41 Learned Shape Gain/Harm Gate

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- positive domains: `['ETH_UCY', 'TrajNet']`
- two-domain gain gate: `True`

| domain | all ADE | t50 ADE | t100 ADE | hard ADE | easy | shape gain all/t50/t100/hard | shape switch | delta shape all/t100 | pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| `ETH_UCY` | 0.0164 | 0.0018 | 0.0043 | 0.0162 | 0.0000 | 0.000679/-0.000126/-0.000044/0.000693 | 0.005926 | 0.000674/-0.000059 | `True` |
| `TrajNet` | 0.0381 | 0.0269 | 0.0138 | 0.0392 | 0.0000 | 0.000117/0.000469/0.000000/0.000129 | 0.000824 | -0.000109/-0.000718 | `True` |

## Interpretation

- This experiment replaces the previous residual-norm heuristic with a train-fitted gain/harm gate for learned waypoint-shape interventions.
- Future waypoint labels are used only to train/evaluate the gain/harm gate; inference inputs remain past-only neural predictions and causal features.
- A positive delta versus the previous learned-shape bridge is needed before calling this a stronger shape-dynamics contribution.
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'val_selected_policy': True, 'stage5c_executed': False, 'smc_enabled': False}`
- claim boundary: `{'learned_shape_gain_harm_gate': True, 'learned_waypoint_shape_residual': True, 'protected_by_endpoint_bridge_or_floor': True, 'latent_generative_rollout': False, 'metric_or_seconds_claim': False, 'true_3d': False, 'foundation_world_model': False}`
