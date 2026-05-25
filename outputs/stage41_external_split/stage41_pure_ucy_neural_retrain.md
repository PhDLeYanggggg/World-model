# Stage41 Strict Pure-UCY Neural Retrain

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- best trial: `pure_ucy_transformer`
- best mode: `bounded_endpoint_residual`
- strict pure UCY-only neural retrain/select/test gate: `True`
- remaining blocker: ``

## Best Test Metrics

- all improvement: `0.090083`
- t50 improvement: `0.088009`
- t100 diagnostic improvement: `0.083106`
- hard/failure improvement: `0.093584`
- easy degradation: `0.000000`
- switch rate: `0.792726`
- neural endpoint without fallback: `{'rows': 35441, 'all_improvement': 0.09230014952008259, 't10_improvement': 0.2945080639312109, 't25_improvement': -0.31497445367685795, 't50_improvement': 0.21303079289629967, 't100_improvement': 0.06383108641982482, 'hard_failure_improvement': 0.1499403484153744, 'easy_degradation': 1.6257516010806778, 'harm_over_fallback': -0.07364632580896721, 'switch_rate': 0.0, 'regret_to_oracle': 0.12474498836204391, 'by_domain': {'ETH_UCY': {'rows': 25901, 'all_improvement': 0.04621994621154035, 't50_improvement': 0.1871286638125077, 't100_improvement': 0.0026131006895112607, 'hard_failure_improvement': 0.11285089841042761, 'easy_degradation': 2.3738767004916275, 'switch_rate': 0.0}, 'UCY': {'rows': 9540, 'all_improvement': 0.21280842523236154, 't50_improvement': 0.2774538691415518, 't100_improvement': 0.2389903817823379, 'hard_failure_improvement': 0.2502984382542832, 'easy_degradation': 0.39570304933311595, 'switch_rate': 0.0}}}`
- neural candidate without fallback: `None`

## Trial Table

| trial | best mode | all | t50 | t100 | hard/failure | easy | switch | strict gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pure_ucy_transformer` | `bounded_endpoint_residual` | 0.0901 | 0.0880 | 0.0831 | 0.0936 | 0.0000 | 0.7927 | `True` |
| `pure_ucy_t50_hard_transformer` | `bounded_endpoint_residual` | 0.0601 | 0.0538 | 0.0280 | 0.0603 | 0.0000 | 0.7823 | `True` |
| `pure_ucy_hybrid_jepa` | `bounded_endpoint_residual` | 0.0827 | 0.0859 | 0.0528 | 0.0844 | 0.0000 | 0.7795 | `True` |

## Dataset

- splits: `{'train': {'rows': 117808, 'sources': 2, 't10': 34363, 't25': 31735, 't50': 29112, 't100': 22598, 'hard': 85739, 'easy': 38264, 'failure': 40518, 'history_len_mean': 23.249082565307617, 'history_ge_32': 31247, 'history_ge_64': 14985}, 'val': {'rows': 16103, 'sources': 1, 't10': 4580, 't25': 4284, 't50': 3988, 't100': 3251, 'hard': 14187, 'easy': 1352, 'failure': 7631, 'history_len_mean': 19.324100494384766, 'history_ge_32': 2251, 'history_ge_64': 517}, 'test': {'rows': 35441, 'sources': 2, 't10': 10283, 't25': 9523, 't50': 8762, 't100': 6873, 'hard': 28250, 'easy': 8193, 'failure': 14776, 'history_len_mean': 21.78970718383789, 'history_ge_32': 7203, 'history_ge_64': 4273}}`
- train-only strongest floor: `{10: 0, 25: 2, 50: 2, 100: 1}`
- source inventory: `{'train': [{'source': 'UCY/students01/students001-trajnet.txt', 'rows': 47223, 't10': 15147, 't25': 13365, 't50': 11583, 't100': 7128, 'is_pure_ucy_source': True}, {'source': 'UCY/students03/obsmat.txt', 'rows': 70585, 't10': 19216, 't25': 18370, 't50': 17529, 't100': 15470, 'is_pure_ucy_source': True}], 'val': [{'source': 'UCY/zara01/obsmat.txt', 'rows': 16103, 't10': 4580, 't25': 4284, 't50': 3988, 't100': 3251, 'is_pure_ucy_source': True}], 'test': [{'source': 'UCY/zara02/obsmat.txt', 'rows': 25901, 't10': 7223, 't25': 6823, 't50': 6422, 't100': 5433, 'is_pure_ucy_source': True}, {'source': 'UCY/zara03/crowds_zara03.txt', 'rows': 9540, 't10': 3060, 't25': 2700, 't50': 2340, 't100': 1440, 'is_pure_ucy_source': True}], 'unused': []}`

## No Leakage

`{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'candidate_fde_input': False, 'candidate_fde_label_only': True, 'train_only_floor_selection': True, 'train_only_normalization_statistics': True, 'central_velocity': False, 'test_endpoint_goals': False, 'stage5c_executed': False, 'smc_enabled': False, 'validation_policy_selection_only': True, 'test_threshold_tuning': False, 'mixed_external_neural_proposal_used': False, 'mixed_external_floor_used': False}`

This is strict UCY-source neural retraining with train-only floor selection and train-only normalization. Future endpoints and candidate FDE are labels/evaluation only.
