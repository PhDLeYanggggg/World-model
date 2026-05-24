# Stage38 Final Report

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / unverified weak metric diagnostic 2.5D world-state selector/correction track。
- SDD / external horizons remain raw-frame horizons, not seconds-level claims.
- Stage5C executed: `False`
- SMC enabled: `False`

## What Was Run

- Froze the Stage35 non-t50 + Stage37 t50 deployable policy.
- Audited UCY, ETH_UCY, TrajNet, and OpenTraj_mixed coverage under the frozen external split.
- Evaluated frozen Stage37 transfer per available external domain.
- Trained bounded correction variants under Stage37 fallback.
- Recomputed 2000-sample bootstrap evidence for frozen Stage37.

## Main Results

- Stage37 frozen all improvement: `0.1348254070727205`
- Stage37 frozen t50 improvement: `0.08457292542209705`
- Stage37 frozen hard/failure improvement: `0.1554340386904196`
- Stage37 frozen easy degradation: `0.0004114683717719725`
- Stage38 correction deployment decision: `keep_stage37_selector`
- Stage38 positive external domains: `['UCY']`
- Stage38 bootstrap evidence: `{'bootstrap_n': 2000, 'all_ci': {'source': 'fresh_run', 'method': 'bootstrap_rows_2000', 'rows': 66303, 'low': 0.13111496554415888, 'mid': 0.1348989904010876, 'high': 0.1387783863042381}, 't50_ci': {'source': 'fresh_run', 'method': 'bootstrap_rows_2000', 'rows': 16263, 'low': 0.07673594462414812, 'mid': 0.08445723096143776, 'high': 0.09207560630337992}, 'hard_failure_ci': {'source': 'fresh_run', 'method': 'bootstrap_rows_2000', 'rows': 45917, 'low': 0.15096500703398077, 'mid': 0.15554037466048337, 'high': 0.15983416236440717}, 'easy_ci': {'source': 'fresh_run', 'method': 'bootstrap_rows_2000', 'rows': 20798, 'low': -0.0009821007527392512, 'mid': -0.00040901792014602645, 'high': 0.0}, 'all': 0.1348254070727205, 'hard_failure': 0.1554340386904196, 'easy_degradation': 0.0004114683717719725}`

## Interpretation

- Stage37 remains the current external best deployable policy.
- Bounded correction showed a t50 diagnostic lift but lost too much all/hard performance, so it is not deployable.
- UCY is positive; ETH/TrajNet are not claimed as successful because held-out external tests are blockers under the frozen split.
- t100 remains diagnostic with no safe improvement.

- gates: `14 / 15`
- verdict: `stage38_robustness_partial_keep_stage37_selector`
