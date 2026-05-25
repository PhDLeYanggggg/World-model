# Stage41 Weighted Pairwise Shape Switch Policy

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- positive domains: `['ETH_UCY', 'TrajNet']`
- two-domain weighted pairwise gate: `True`
- better than fixed composer on any core metric: `['TrajNet']`

| domain | all ADE | t50 ADE | t100 ADE | hard ADE | easy | shape gain all/t50/t100/hard | sign acc | fixed delta all/t50/t100/hard | pass |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| `ETH_UCY` | 0.0161 | 0.0018 | 0.0043 | 0.0159 | 0.0000 | 0.000369/-0.000126/-0.000030/0.000384 | 0.3790 | -0.000356/-0.000125/-0.000044/-0.000358 | `True` |
| `TrajNet` | 0.0383 | 0.0267 | 0.0145 | 0.0394 | 0.0000 | 0.000282/0.000223/0.000718/0.000309 | 0.5065 | 0.000158/-0.000239/0.000708/0.000173 | `True` |

## Interpretation

- This experiment addresses the previous pairwise model's rare-positive switch labels by upweighting hard/failure, t50/t100, source-switch, and positive-gain rows during training.
- The hard/failure labels are training weights only and are not available as inference inputs.
- Validation selects conservative deployment thresholds; test is evaluated once.
- This remains protected dataset-local raw-frame 2.5D evidence, not Stage5C/SMC/metric/seconds/true-3D/foundation evidence.
- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'hard_failure_labels_inference_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'val_selected_policy': True, 'stage5c_executed': False, 'smc_enabled': False}`
- claim boundary: `{'weighted_pairwise_gain_harm_source_switch': True, 'learned_waypoint_shape_residual': True, 'protected_by_endpoint_bridge_or_floor': True, 'latent_generative_rollout': False, 'metric_or_seconds_claim': False, 'true_3d': False, 'foundation_world_model': False}`
