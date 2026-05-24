# Stage30 Final Report

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space benchmark，不是 metric benchmark。
- t+50 / t+100 是 raw-frame horizon，不能说成 seconds-level。
- homography / scale / effective seconds 未验证，除非 Stage30 raw audit 证明。
- Stage5C 未执行。
- SMC 未启用。

- Stage5C executed: `False`
- SMC enabled: `False`

- t+50 fresh recheck: `0.1686288243790961`
- hard/failure fresh recheck: `0.1336398986813968`
- easy degradation fresh recheck: `0.01928694490688554`
- 3000 bootstrap t+50: `{'mean': 0.16848943145890402, 'std': 0.004304755428418378, 'ci_low': 0.16009449184493582, 'ci_high': 0.17713789303073652, 'n': 24810}`
- external conversion status: `converted_diagnostic_non_sdd`
- external transfer eval: `{'status': 'not_run', 'reason': 'Full M3W-LAS all_latent transfer needs external latent cache and scale calibration; converted base feature store is diagnostic only.'}`
- time/geometry conclusion: `pixel raw-frame only`
- world model capability: `{'source': 'fresh_run', 'is_selector_only_trick': 'partly_selector_policy_but_latent_features_improve_selector_decisions', 'latent_t50_delta': 0.059488896481791745, 'goal_t50_delta': 0.00011240042697294172, 'interaction_hard_delta': 0.00021180243131857512, 'subset_contribution_probe': {'source': 'fresh_run', 'seed': 0, 'subsets': {'all': {'n': 100000, 'goal_delta': 0.0002045059207706882, 'interaction_delta': 0.0005745775759933341}, 't50': {'n': 24810, 'goal_delta': 0.00022173077274224414, 'interaction_delta': 0.0008041296522258143}, 'hard': {'n': 96581, 'goal_delta': 0.00020220240594427213, 'interaction_delta': 0.0005905242316518708}, 'high_density': {'n': 26945, 'goal_delta': 3.837264599267089e-05, 'interaction_delta': 0.0014998137488503567}}}, 'interaction_high_density_delta': 0.0014998137488503567, 'contribution_threshold': 0.001, 'latent_contribution': True, 'goal_contribution': False, 'interaction_contribution': True, 'interaction_contribution_scope': 'high_density_subset', 'cross_scene_stable': True, 'external_generalization': False, 'external_blocker': 'Full M3W-LAS all_latent transfer needs external latent cache and scale calibration; converted base feature store is diagnostic only.', 'still_sdd_pixel_candidate': True, 'stage5c_executed': False, 'smc_enabled': False}`
- gates: `14 / 14`
- tests: `python -m pytest tests` -> `54 passed`
- verdict: `stage30_fresh_recompute_verified_m3w_las_v2_candidate_not_stage5c_ready`
