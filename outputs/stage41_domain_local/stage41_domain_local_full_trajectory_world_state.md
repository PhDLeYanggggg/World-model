# Stage41 Domain-Local Learned Full-Trajectory World-State Audit

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- domains run: `['ETH_UCY', 'TrajNet']`
- blockers: `{'UCY': {'status': 'not_run', 'reason': 'domain lacks strict train/val/test coverage in current all-agent split', 'has_train': True, 'has_val': False, 'has_test': True}}`
- positive domains: `[]`
- two-domain full-trajectory gate: `False`
- failure taxonomy: `{'ETH_UCY': {'reasons': ['t50_ade_not_positive', 't100_ade_not_positive', 'same_frame_proximity_delta_unsafe', 'ungated_neural_catastrophic_easy_degradation'], 'ade_all': 0.03200185360974939, 'ade_t50': 0.0, 'ade_t100': 0.0, 'fde_all': 0.02140015213054336, 'fde_t50': 0.0, 'collision_delta_vs_floor_005': 0.020563149900689304, 'next_fix': 'train a validation-selected efficient proximity guard and horizon-specific waypoint policy; do not deploy learned full-waypoint domain-local model yet.'}, 'TrajNet': {'reasons': ['all_ade_not_positive', 't50_ade_not_positive', 'hard_failure_ade_not_positive', 'same_frame_proximity_delta_unsafe', 'ungated_neural_catastrophic_easy_degradation', 'endpoint_fde_positive_but_waypoint_ade_negative'], 'ade_all': -0.009876103097439914, 'ade_t50': -0.09471803973274517, 'ade_t100': 0.022840260699697912, 'fde_all': 0.06300366873346075, 'fde_t50': 0.06502418472340132, 'collision_delta_vs_floor_005': 0.014909478168264101, 'next_fix': 'train a validation-selected efficient proximity guard and horizon-specific waypoint policy; do not deploy learned full-waypoint domain-local model yet.'}}`

| domain | train/val/test rows | all | t50 | t100 | hard | easy | multi all | collision d005 | pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ETH_UCY` | `108794/16103/25901` | 0.0320 | 0.0000 | 0.0000 | 0.0344 | 0.0000 | 0.0324 | 0.0206 | `False` |
| `TrajNet` | `63650/37153/20087` | -0.0099 | -0.0947 | 0.0228 | -0.0111 | 0.0000 | -0.0100 | 0.0149 | `False` |

- no leakage: `{'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'val_selected_policy': True, 'stage5c_executed': False, 'smc_enabled': False}`
- claim boundary: `{'learned_full_waypoint_neural_dynamics': True, 'all_active_agent_world_state_audit': True, 'latent_generative_rollout': False, 'metric_or_seconds_claim': False, 'true_3d': False, 'foundation_world_model': False}`
