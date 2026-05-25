# Stage41 Joint Residual Domain-Horizon Policy Repair

- source: `fresh_run`
- selected trial: `joint_residual_clip100_balanced`
- deployable: `False`
- test metrics: `{'rows': 55528, 'all_improvement': -0.0006363835790781369, 't10_improvement': 0.0, 't25_improvement': -0.002924270188935596, 't50_improvement': 0.0, 't100_improvement': -0.0005775775254206472, 'hard_failure_improvement': -0.0005725919532908463, 'easy_degradation': 0.00111839385467416, 'harm_over_fallback': 0.0003045921730170449, 'switch_rate': 0.0020530182970753493, 'regret_to_oracle': -0.02080476636729034, 'by_domain': {'ETH_UCY': {'rows': 25901, 'all_improvement': -0.0006433948520336852, 't50_improvement': 0.0, 't100_improvement': -0.0015654796439141805, 'hard_failure_improvement': -0.0006912265741825241, 'easy_degradation': 0.0, 'switch_rate': 0.0006177367669201961}, 'TrajNet': {'rows': 20087, 'all_improvement': -0.0010350309954842984, 't50_improvement': 0.0, 't100_improvement': 0.0009135985925908807, 'hard_failure_improvement': -0.0007297772400616243, 'easy_degradation': 0.0030227625880583364, 'switch_rate': 0.004878777318663812}, 'UCY': {'rows': 9540, 'all_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'switch_rate': 0.0}}}`
- test switch rate: `0.0020530182970753493`
- test collision delta @0.05 normalized: `0.0008621005906305768`
- no leakage: `{'future_waypoints_input': False, 'future_waypoints_label_eval_only': True, 'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'policy_selected_on_val': True, 'stage5c_executed': False, 'smc_enabled': False}`

The policy is selected only on validation slices. Domains/horizons without validation support fall back to the floor policy.
