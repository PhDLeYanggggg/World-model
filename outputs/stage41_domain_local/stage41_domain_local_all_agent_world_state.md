# Stage41 Domain-Local All-Agent World-State Audit

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- positive domains: `['ETH_UCY', 'UCY_expanded']`
- two-domain all-agent world-state gate: `True`

| domain | rows train/val/test | ADE all | ADE t50 | ADE t100 | ADE hard | easy | multi all | multi t50 | collision d005 | pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `ETH_UCY` | `41501/4648/21598` | 0.0199 | 0.0036 | 0.0038 | 0.0196 | 0.0000 | 0.0205 | 0.0037 | 0.0040 | `True` |
| `TrajNet` | `35009/6098/3639` | 0.0555 | 0.0628 | 0.0340 | 0.0577 | 0.0000 | 0.0589 | 0.0688 | 0.0134 | `False` |
| `UCY` | `3490/13254/9540` | -0.0043 | -0.0067 | 0.0000 | -0.0002 | 0.1602 | -0.0038 | -0.0061 | 0.0046 | `False` |
| `UCY_expanded` | `117808/16103/35441` | 0.0857 | 0.1372 | 0.0247 | 0.0903 | 0.0000 | 0.0868 | 0.1384 | -0.0019 | `True` |

This audit deliberately uses endpoint-linear waypoints so that a domain-local endpoint neural model can be checked against same-frame all-agent proximity and waypoint ADE/FDE. It is not claimed as a learned full-waypoint neural rollout.

- no leakage: `{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'val_selected_policy': True, 'stage5c_executed': False, 'smc_enabled': False}`
- claim boundary: `{'domain_local_neural_endpoint_retrain': True, 'linear_endpoint_waypoint_proxy': True, 'learned_full_waypoint_rollout': False, 'latent_generative_rollout': False, 'metric_or_seconds_claim': False, 'true_3d': False, 'foundation_world_model': False}`
