# Stage 13 Search Results

- executed_training: `True`
- trial_count: `24`
- episode_count: `660`
- latent_enabled: `False`
- smc_enabled: `False`

## Best Summary

- best_all_test: `{'model': 'residual_no_alpha', 'dataset': 'eth_ucy_ewap', 'subset': 'all', 'horizon': 50, 'FDE': 2.784152, 'ADE': 2.784152, 'baseline_FDE': 2.821185, 'baseline_ADE': 2.821185, 'improvement': 0.013127, 'episodes': 21, 'alpha': 1.0, 'residual_magnitude': 0.1, 'physical_validity': 0.9, 'trial_id': 4, 'alpha_regularization': 'medium', 'residual_clip': 0.1, 'hard_failure_weight': 1, 't100_weight': 1, 'interaction_mode': 'none', 'scene_goal_mode': 'none'}`
- best_eth_ucy_ewap_t100: `None`
- best_hard: `{'model': 'residual_no_alpha', 'dataset': 'eth_ucy_ewap', 'subset': 'hard', 'horizon': 50, 'FDE': 2.784152, 'ADE': 2.784152, 'baseline_FDE': 2.821185, 'baseline_ADE': 2.821185, 'improvement': 0.013127, 'episodes': 21, 'alpha': 1.0, 'residual_magnitude': 0.1, 'physical_validity': 0.9, 'trial_id': 4, 'alpha_regularization': 'medium', 'residual_clip': 0.1, 'hard_failure_weight': 1, 't100_weight': 1, 'interaction_mode': 'none', 'scene_goal_mode': 'none'}`
- best_baseline_failure: `{'model': 'residual_no_alpha', 'dataset': 'eth_ucy_ewap', 'subset': 'baseline_failure', 'horizon': 50, 'FDE': 2.784152, 'ADE': 2.784152, 'baseline_FDE': 2.821185, 'baseline_ADE': 2.821185, 'improvement': 0.013127, 'episodes': 21, 'alpha': 1.0, 'residual_magnitude': 0.1, 'physical_validity': 0.9, 'trial_id': 4, 'alpha_regularization': 'medium', 'residual_clip': 0.1, 'hard_failure_weight': 1, 't100_weight': 1, 'interaction_mode': 'none', 'scene_goal_mode': 'none'}`
- best_easy_preservation: `{'model': 'alpha_only_no_residual', 'dataset': 'eth_ucy', 'subset': 'easy', 'horizon': 1, 'FDE': 0.410458, 'ADE': 0.410458, 'baseline_FDE': 0.410458, 'baseline_ADE': 0.410458, 'improvement': 0.0, 'episodes': 3, 'alpha': 0.0, 'residual_magnitude': 0.0, 'physical_validity': 1.0, 'trial_id': 1, 'alpha_regularization': 'low', 'residual_clip': 0.05, 'hard_failure_weight': 1, 't100_weight': 1, 'interaction_mode': 'none', 'scene_goal_mode': 'none'}`

This is deterministic repair search only. It does not authorize Stage 5C or SMC.
