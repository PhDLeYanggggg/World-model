# Stage 13 Search Results

- executed_training: `True`
- trial_count: `10`
- episode_count: `724`
- latent_enabled: `False`
- smc_enabled: `False`

## Best Summary

- best_all_test: `{'model': 'scene_interaction', 'dataset': 'eth_ucy_ewap_stage14', 'subset': 'all', 'horizon': 1, 'FDE': 0.583899, 'ADE': 0.583899, 'baseline_FDE': 0.60757, 'baseline_ADE': 0.60757, 'improvement': 0.038961, 'episodes': 13, 'alpha': 0.75, 'residual_magnitude': 0.0375, 'physical_validity': 0.9625, 'trial_id': 9, 'alpha_regularization': 'low', 'residual_clip': 0.05, 'hard_failure_weight': 1, 't100_weight': 1, 'interaction_mode': 'none', 'scene_goal_mode': 'none'}`
- best_eth_ucy_ewap_t100: `{'model': 'residual_no_alpha', 'dataset': 'eth_ucy_ewap_stage14', 'subset': 'all', 'horizon': 100, 'FDE': 5.875909, 'ADE': 5.875909, 'baseline_FDE': 5.923605, 'baseline_ADE': 5.923605, 'improvement': 0.008052, 'episodes': 13, 'alpha': 1.0, 'residual_magnitude': 0.05, 'physical_validity': 0.95, 'trial_id': 2, 'alpha_regularization': 'low', 'residual_clip': 0.05, 'hard_failure_weight': 1, 't100_weight': 1, 'interaction_mode': 'none', 'scene_goal_mode': 'none'}`
- best_hard: `{'model': 'alpha_only_no_residual', 'dataset': 'trajnet', 'subset': 'hard', 'horizon': 1, 'FDE': 0.291144, 'ADE': 0.291144, 'baseline_FDE': 0.291144, 'baseline_ADE': 0.291144, 'improvement': 0.0, 'episodes': 30, 'alpha': 0.0, 'residual_magnitude': 0.0, 'physical_validity': 1.0, 'trial_id': 1, 'alpha_regularization': 'low', 'residual_clip': 0.05, 'hard_failure_weight': 1, 't100_weight': 1, 'interaction_mode': 'none', 'scene_goal_mode': 'none'}`
- best_baseline_failure: `{'model': 'alpha_only_no_residual', 'dataset': 'trajnet', 'subset': 'baseline_failure', 'horizon': 1, 'FDE': 0.291144, 'ADE': 0.291144, 'baseline_FDE': 0.291144, 'baseline_ADE': 0.291144, 'improvement': 0.0, 'episodes': 30, 'alpha': 0.0, 'residual_magnitude': 0.0, 'physical_validity': 1.0, 'trial_id': 1, 'alpha_regularization': 'low', 'residual_clip': 0.05, 'hard_failure_weight': 1, 't100_weight': 1, 'interaction_mode': 'none', 'scene_goal_mode': 'none'}`
- best_easy_preservation: `{'model': 'scene_interaction', 'dataset': 'eth_ucy_ewap_stage14', 'subset': 'easy', 'horizon': 1, 'FDE': 0.583899, 'ADE': 0.583899, 'baseline_FDE': 0.60757, 'baseline_ADE': 0.60757, 'improvement': 0.038961, 'episodes': 13, 'alpha': 0.75, 'residual_magnitude': 0.0375, 'physical_validity': 0.9625, 'trial_id': 9, 'alpha_regularization': 'low', 'residual_clip': 0.05, 'hard_failure_weight': 1, 't100_weight': 1, 'interaction_mode': 'none', 'scene_goal_mode': 'none'}`

This is deterministic repair search only. It does not authorize Stage 5C or SMC.
