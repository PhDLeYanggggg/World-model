# Stage40 Final Report

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- external/SDD remain raw-frame dataset-local or pixel-space; no metric/seconds claim.
- Stage5C executed: `False`; SMC enabled: `False`.

## Direct Answers

- 是否训练了神经世界模型: `是`
- 是否超过 Stage37: `False`
- 是否部署 neural: `False`
- 如果没有，为什么: `neural trials did not beat Stage37 on same-subset all/t50/hard under easy<=2%; Stage37 remains stronger and safer`
- 当前 best deployable: `Stage37 selector`
- 距离真正 world model: 需要跨 ETH/TrajNet held-out split、t100 lift、神经 dynamics 在 Stage37 之外稳定提供可部署提升。

## Best Stage40 Result

- best neural: `Stage40_causal_transformer_candidate_ranker`
- best metrics: `{'rows': 16000, 'all_improvement': 0.13200527968500975, 't10_improvement': 0.3025072845476737, 't25_improvement': 0.0, 't50_improvement': 0.08301228991324938, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1523895057047563, 'easy_degradation': 0.0006815189764808327, 'harm_over_fallback': -0.13887056636996567, 'switch_rate': 0.0, 'neural_without_fallback': {'rows': 16000, 'all_improvement': -1.2636329485667623, 't10_improvement': -0.6609946384475607, 't25_improvement': -1.7309633497217503, 't50_improvement': -2.921009455663605, 't100_improvement': -0.6466884359881155, 'hard_failure_improvement': -1.093965878739179, 'easy_degradation': 6.123116549641697, 'harm_over_fallback': 1.3293515507103102}}`
- Stage37 reference: `{'rows': 16000, 'all_improvement': 0.13200527968500975, 't10_improvement': 0.3025072845476737, 't25_improvement': 0.0, 't50_improvement': 0.08301228991324938, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1523895057047563, 'easy_degradation': 0.0006815189764808327, 'harm_over_fallback': -0.13887056636996567}`
- gates: `11 / 12`
- verdict: `stage40_neural_optimization_keep_stage37`
