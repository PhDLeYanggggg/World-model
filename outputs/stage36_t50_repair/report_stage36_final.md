# Stage36 Final Report

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。
- External coordinates remain dataset-local / unverified weak metric diagnostic.
- t+50 / t+100 remain raw-frame horizons; t+100 is diagnostic.
- Stage5C executed: `False`
- SMC enabled: `False`

## Outcome

- Stage35 external t+50 rows: `16263`
- t+50 oracle headroom: `0.22982786182653314`
- final external metrics: `{'rows': 66303, 'all_improvement': 0.12131890857784355, 't10_improvement': 0.30620476404756647, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1398494448930071, 'easy_degradation': 0.0004114683717719725, 'selector_regret': 0.4289430569144169, 'harm_over_fallback': -0.12772804655228734, 'switch_rate': 0.04999773765892946, 'mean_confidence': 0.03720055893063545, 'status': 'fresh_run', 'reason': 'Stage36 final external policy; t50 remains a hard gate.'}`
- gates: `12 / 14`
- verdict: `stage36_t50_transfer_not_repaired`

## Interpretation

- Stage36 did not repair the t+50 gate.
- all-test and hard/failure remain positive via Stage35-style selective transfer, but the long-horizon t+50 slice stays at `0.0` improvement.
- Because t+50 is the explicit Stage36 deployment blocker, this is not a deployable cross-domain M3W candidate.
- failure analysis: `t50 selector has oracle headroom but insufficient reliable causal features/goal context to pass the >3% t50 deployment gate.`
