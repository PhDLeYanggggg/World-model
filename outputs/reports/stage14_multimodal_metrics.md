# Stage 14 Multimodal Benchmark

- metric_rows: `940`
- EWAP t+100 mask can evaluate: `True`
- best eth_ucy_ewap t+100: `{'model': 'residual_no_alpha', 'dataset': 'eth_ucy_ewap_stage14', 'subset': 'all', 'horizon': 100, 'FDE': 5.875909, 'ADE': 5.875909, 'baseline_FDE': 5.923605, 'baseline_ADE': 5.923605, 'improvement': 0.008052, 'episodes': 13, 'alpha': 1.0, 'residual_magnitude': 0.05, 'physical_validity': 0.95, 'trial_id': 2, 'alpha_regularization': 'low', 'residual_clip': 0.05, 'hard_failure_weight': 1, 't100_weight': 1, 'interaction_mode': 'none', 'scene_goal_mode': 'none'}`
- best HardBench: `{'model': 'alpha_only_no_residual', 'dataset': 'trajnet', 'subset': 'hard', 'horizon': 1, 'FDE': 0.291144, 'ADE': 0.291144, 'baseline_FDE': 0.291144, 'baseline_ADE': 0.291144, 'improvement': 0.0, 'episodes': 30, 'alpha': 0.0, 'residual_magnitude': 0.0, 'physical_validity': 1.0, 'trial_id': 1, 'alpha_regularization': 'low', 'residual_clip': 0.05, 'hard_failure_weight': 1, 't100_weight': 1, 'interaction_mode': 'none', 'scene_goal_mode': 'none'}`
- best BaselineFailureBench: `{'model': 'alpha_only_no_residual', 'dataset': 'trajnet', 'subset': 'baseline_failure', 'horizon': 1, 'FDE': 0.291144, 'ADE': 0.291144, 'baseline_FDE': 0.291144, 'baseline_ADE': 0.291144, 'improvement': 0.0, 'episodes': 30, 'alpha': 0.0, 'residual_magnitude': 0.0, 'physical_validity': 1.0, 'trial_id': 1, 'alpha_regularization': 'low', 'residual_clip': 0.05, 'hard_failure_weight': 1, 't100_weight': 1, 'interaction_mode': 'none', 'scene_goal_mode': 'none'}`
- visual/raster scene gain: `0.0`
- scene/goal gain: `0.0`
- interaction gain: `0.0`

This benchmark is deterministic only. Latent generative modeling and SMC remain disabled.
