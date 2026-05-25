# Stage41 Bounded Neural Blend Dynamics

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- selected policy: `{'type': 'global', 'alpha': 0.3}`
- deployable: `False`
- non-fallback continuous neural contribution: `False`
- test metrics: `{'rows': 55528, 'all_improvement': 0.183054549856548, 't10_improvement': 0.21785183503121108, 't25_improvement': 0.0988494282238217, 't50_improvement': 0.17556642701259895, 't100_improvement': 0.1988123052757771, 'hard_failure_improvement': 0.1934724604647473, 'easy_degradation': 0.2070880438160938, 'harm_over_fallback': -0.08761537059496428, 'switch_rate': 1.0, 'regret_to_oracle': -0.10872472913527169, 'by_domain': {'ETH_UCY': {'rows': 25901, 'all_improvement': 0.16687241917968965, 't50_improvement': 0.1765012637084158, 't100_improvement': 0.15487846000150585, 'hard_failure_improvement': 0.17343328261369984, 'easy_degradation': 0.20487482757010755, 'switch_rate': 1.0}, 'TrajNet': {'rows': 20087, 'all_improvement': 0.19226758347573192, 't50_improvement': 0.17165948219733262, 't100_improvement': 0.24661785078534038, 'hard_failure_improvement': 0.21280476888494493, 'easy_degradation': 0.34435364639297195, 'switch_rate': 1.0}, 'UCY': {'rows': 9540, 'all_improvement': 0.2118546423683053, 't50_improvement': 0.1792327058440344, 't100_improvement': 0.25256033703041125, 'hard_failure_improvement': 0.219121192266032, 'easy_degradation': 0.0, 'switch_rate': 1.0}}, 'alpha_mean': 0.2999999999999999, 'alpha_positive_rate': 1.0, 'collision_delta_vs_floor_005': 0.007905645841740305, 'smoothness_jagged_delta': 0.0}`
- no leakage: `{'future_endpoint_input': False, 'future_labels_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False, 'test_threshold_tuning': False, 'blend_policy_selected_on_val': True, 'stage5c_executed': False, 'smc_enabled': False}`

This evaluates a validation-selected bounded blend `floor + alpha * (neural - floor)` across rows. It is a conservative neural dynamics head, not Stage5C latent generation.
