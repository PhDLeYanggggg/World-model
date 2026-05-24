# Stage32 Final Report

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- External coordinates are dataset-local / unverified weak metric diagnostic; no metric seconds claim.
- Stage5C executed: `False`
- SMC enabled: `False`

- best adapted selector metrics: `{'domain': 'external', 'split': 'test', 'rows': 3636, 'all_improvement': 0.0, 't10_improvement': 0.0, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'selector_regret': 0.01730417331328272, 'harm_over_fallback': 0.0, 'switch_rate': 0.0, 'mean_confidence': 0.0}`
- cross-domain matrix summary keys: `['SDD_train_to_SDD_test', 'SDD_train_to_external_test', 'external_train_to_external_test', 'external_train_to_SDD_test', 'SDD_external_train_to_SDD_test', 'SDD_external_train_to_external_test']`
- domain failure status: `not_cross_domain_candidate`
- gates: `9 / 11`
- verdict: `stage32_domain_alignment_partial_not_cross_domain_candidate`
