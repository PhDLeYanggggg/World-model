# Stage41 Domain-Local Full-Trajectory Repair

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- trajectory modes tested: `['raw_waypoint', 'endpoint_linearized', 'blend_25_raw_75_endpoint']`
- positive domains: `['TrajNet']`
- two-domain repair gate: `False`
- failure taxonomy: `{'ETH_UCY': {'reasons': ['t50_ade_not_positive', 'easy_degradation_over_2pct'], 'trajectory_mode': 'blend_25_raw_75_endpoint', 'policy_family': 'gain_calibrated', 'deployment_variant': 'long_horizon_only', 'ade_all': 0.011436216574168045, 'ade_t50': 0.0, 'ade_t100': 0.027826091852716672, 'fde_all': 0.008311019616787152, 'fde_t50': 0.0, 'collision_delta_vs_floor_005': 0.001207306149472287}}`

| domain | family | variant | mode | all ADE | t50 ADE | t100 ADE | hard ADE | easy | multi all | collision d005 | guard off | pass |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ETH_UCY` | `gain_calibrated` | `long_horizon_only` | `blend_25_raw_75_endpoint` | 0.0114 | 0.0000 | 0.0278 | 0.0123 | 0.0304 | 0.0128 | 0.0012 | 0 | `False` |
| `TrajNet` | `gain_calibrated` | `t50_only` | `raw_waypoint` | 0.0028 | 0.0109 | 0.0000 | 0.0031 | 0.0040 | 0.0029 | -0.0005 | 0 | `True` |

## Interpretation

- This repair uses cached trained neural full-waypoint checkpoints and performs a fresh validation-selected policy/evaluation pass.
- Endpoint-linearized mode is explicitly labeled when selected: it tests whether the neural model learned endpoint dynamics while failing intermediate waypoint shape. It is not claimed as learned full-shape dynamics.
- Gain-calibrated mode trains a ridge switch head on train split labels only, selects thresholds on val, and evaluates test once.
- Inference-time switching does not consume hard/easy/failure labels; those remain validation/evaluation labels.
- no leakage: `{'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'val_selected_policy': True, 'inference_switch_uses_hard_easy_labels': False, 'stage5c_executed': False, 'smc_enabled': False}`
- claim boundary: `{'learned_full_waypoint_neural_dynamics': True, 'endpoint_linearized_repair_if_selected': True, 'all_active_agent_world_state_audit': True, 'latent_generative_rollout': False, 'metric_or_seconds_claim': False, 'true_3d': False, 'foundation_world_model': False}`
