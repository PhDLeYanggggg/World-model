# Stage41 Final Report

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D multi-agent trajectory world-state model.
- External/SDD remain raw-frame dataset-local or pixel-space; no metric/seconds claim.
- Stage5C executed: `False`; SMC enabled: `False`.

## Direct Answers

- 是否训练了 neural world model: `是`
- 是否超过 Stage37: `False`
- 是否超过 strongest causal baseline: `False`
- 是否有两个以上 external domain 正迁移: `False`
- t50 是否改善: `False`
- t100 是否改善: `False`
- hard/failure 是否改善: `False`
- easy 是否保持: `True`
- JEPA 是否有用: `未证明，除非 hybrid/JEPA trial 在 gates 中胜出`
- Transformer 是否有用: `仅当 best Stage41 trial 过 Stage37 margin gate 才可称有 deployable lift`
- 是否仍只是 2.5D: `是`
- 是否可称 foundation world model: `否`
- 是否可以 Stage5C: `否`
- 是否可以 SMC: `否`
- 当前最强 deployable: `Stage37 selector`

## Best Result

- best Stage41 neural: `conformal_safety_head_transformer`
- best metrics: `{'rows': 34777, 'all_improvement': 0.0, 't10_improvement': 0.0, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'harm_over_fallback': 0.0, 'switch_rate': 0.0, 'regret_to_oracle': 0.5311833439556495, 'by_domain': {'ETH_UCY': {'rows': 21598, 'all_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'switch_rate': 0.0}, 'TrajNet': {'rows': 3639, 'all_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'switch_rate': 0.0}, 'UCY': {'rows': 9540, 'all_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'switch_rate': 0.0}}, 'neural_endpoint_without_fallback': {'rows': 34777, 'all_improvement': 0.03511888712276556, 't10_improvement': 0.4038711381418798, 't25_improvement': -0.3162298418193421, 't50_improvement': -0.18197395372873815, 't100_improvement': 0.1380159267201967, 'hard_failure_improvement': 0.05167724357600689, 'easy_degradation': 0.39921317883150875, 'harm_over_fallback': -0.04949312285282851, 'switch_rate': 0.0, 'regret_to_oracle': 0.481690221102821, 'by_domain': {'ETH_UCY': {'rows': 21598, 'all_improvement': -0.01392309205234521, 't50_improvement': -0.3048523738493607, 't100_improvement': 0.11269159751681213, 'hard_failure_improvement': 0.002830273029555519, 'easy_degradation': 0.2039057447431205, 'switch_rate': 0.0}, 'TrajNet': {'rows': 3639, 'all_improvement': 0.19295072551089476, 't50_improvement': 0.22842751948245177, 't100_improvement': -0.004603809412569104, 'hard_failure_improvement': 0.1846991331776392, 'easy_degradation': 0.10365985055750038, 'switch_rate': 0.0}, 'UCY': {'rows': 9540, 'all_improvement': 0.2321273961173841, 't50_improvement': 0.19109431435167568, 't100_improvement': 0.31441598394746806, 'hard_failure_improvement': 0.27007693279926337, 'easy_degradation': 1.1280521527387366, 'switch_rate': 0.0}}}, 'neural_candidate_without_fallback': {'rows': 34777, 'all_improvement': -0.8484113157715827, 't10_improvement': -0.012814343236110748, 't25_improvement': -1.6072338708009695, 't50_improvement': -1.5111419028000146, 't100_improvement': -0.5210491294120758, 'hard_failure_improvement': -0.8398415355070659, 'easy_degradation': 1.3259754511292763, 'harm_over_fallback': 1.1956678847603077, 'switch_rate': 0.0, 'regret_to_oracle': 1.7268512287159572, 'by_domain': {'ETH_UCY': {'rows': 21598, 'all_improvement': -0.8236572591897515, 't50_improvement': -1.638348429614327, 't100_improvement': -0.4238858856948182, 'hard_failure_improvement': -0.8105521673599834, 'easy_degradation': 0.9244187743422161, 'switch_rate': 0.0}, 'TrajNet': {'rows': 3639, 'all_improvement': -0.8910476940190553, 't50_improvement': -1.0201582404083345, 't100_improvement': -0.9733563608509908, 'hard_failure_improvement': -0.9043529907735715, 'easy_degradation': 2.2373631075916247, 'switch_rate': 0.0}, 'UCY': {'rows': 9540, 'all_improvement': -0.9633287289041492, 't50_improvement': -1.1529334727211995, 't100_improvement': -0.8595388422178205, 'hard_failure_improvement': -0.9772652488001254, 'easy_degradation': 2.114313739303303, 'switch_rate': 0.0}}}, 'selected_candidate_distribution': {'0': 34777}, 't50_ci': {'low': 0.0, 'mid': 0.0, 'high': 0.0, 'n': 8245}, 'hard_failure_ci': {'low': 0.0, 'mid': 0.0, 'high': 0.0, 'n': 29370}}`
- gates: `12 / 17`
- verdict: `stage41_breakthrough_not_yet_keep_stage37`

## Failure / Gap

- failure taxonomy: `{'external_split': 'Rebuilt test now includes multiple domains; exact Stage37 frozen policy was originally validated on UCY-style test and is not identical.', 'world_model_dataset': 'External row cache provides per-agent history plus neighbor aggregates, not complete all-agent token sequences.', 'neural_without_fallback': {'rows': 34777, 'all_improvement': 0.03511888712276556, 't10_improvement': 0.4038711381418798, 't25_improvement': -0.3162298418193421, 't50_improvement': -0.18197395372873815, 't100_improvement': 0.1380159267201967, 'hard_failure_improvement': 0.05167724357600689, 'easy_degradation': 0.39921317883150875, 'harm_over_fallback': -0.04949312285282851, 'switch_rate': 0.0, 'regret_to_oracle': 0.481690221102821, 'by_domain': {'ETH_UCY': {'rows': 21598, 'all_improvement': -0.01392309205234521, 't50_improvement': -0.3048523738493607, 't100_improvement': 0.11269159751681213, 'hard_failure_improvement': 0.002830273029555519, 'easy_degradation': 0.2039057447431205, 'switch_rate': 0.0}, 'TrajNet': {'rows': 3639, 'all_improvement': 0.19295072551089476, 't50_improvement': 0.22842751948245177, 't100_improvement': -0.004603809412569104, 'hard_failure_improvement': 0.1846991331776392, 'easy_degradation': 0.10365985055750038, 'switch_rate': 0.0}, 'UCY': {'rows': 9540, 'all_improvement': 0.2321273961173841, 't50_improvement': 0.19109431435167568, 't100_improvement': 0.31441598394746806, 'hard_failure_improvement': 0.27007693279926337, 'easy_degradation': 1.1280521527387366, 'switch_rate': 0.0}}}, 'fallback_competition': 'Stage37/causal floor is strong; neural must switch sparingly and with calibrated gain/harm.', 't100': 't100 remains raw-frame diagnostic; positive only if metrics show it, otherwise blocker is horizon context/track stability.', 'jepa': 'JEPA is representation auxiliary only; no generative rollout or Stage5C execution.'}`
