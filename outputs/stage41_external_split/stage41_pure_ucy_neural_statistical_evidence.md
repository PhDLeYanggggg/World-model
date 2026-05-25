# Stage41 Strict Pure-UCY Neural Statistical Evidence

- source: `fresh_run`
- Stage5C executed: `False`
- SMC enabled: `False`
- metric/seconds claim: `False`
- best trial/mode: `pure_ucy_transformer` / `bounded_endpoint_residual`
- statistically stable on test: `True`

## Recomputed Test Metrics

- all improvement: `0.09008338741058997`
- t50 improvement: `0.08800918427966675`
- t100 raw-frame diagnostic improvement: `0.08310612104619552`
- hard/failure improvement: `0.09358425348430643`
- easy degradation: `0.0`
- switch rate: `0.792725938884343`

## Bootstrap

- bootstrap n: `2000`
- all/t50/t100/hard lows: `0.08888491295965614` / `0.08630755189944371` / `0.08069804446646152` / `0.09229635491900144`
- by source: `{'UCY/zara02/obsmat.txt': {'low': 0.09673364730105438, 'mid': 0.0982820306175517, 'high': 0.09985136850302054, 'n': 25901, 'bootstrap_n': 2000}, 'UCY/zara03/crowds_zara03.txt': {'low': 0.06707415954927377, 'mid': 0.06857230531591169, 'high': 0.07016195297326545, 'n': 9540, 'bootstrap_n': 2000}}`

## No-Fallback Negative Evidence

- raw neural endpoint without fallback: `{'rows': 35441, 'all_improvement': 0.09230014952008259, 't10_improvement': 0.2945080639312109, 't25_improvement': -0.31497445367685795, 't50_improvement': 0.21303079289629967, 't100_improvement': 0.06383108641982482, 'hard_failure_improvement': 0.1499403484153744, 'easy_degradation': 1.6257516010806778, 'harm_over_fallback': -0.07364632580896721, 'switch_rate': 1.0, 'regret_to_oracle': 0.12474498836204391, 'by_domain': {'ETH_UCY': {'rows': 25901, 'all_improvement': 0.04621994621154035, 't50_improvement': 0.1871286638125077, 't100_improvement': 0.0026131006895112607, 'hard_failure_improvement': 0.11285089841042761, 'easy_degradation': 2.3738767004916275, 'switch_rate': 1.0}, 'UCY': {'rows': 9540, 'all_improvement': 0.21280842523236154, 't50_improvement': 0.2774538691415518, 't100_improvement': 0.2389903817823379, 'hard_failure_improvement': 0.2502984382542832, 'easy_degradation': 0.39570304933311595, 'switch_rate': 1.0}}}`

## No Leakage

`{'future_endpoint_input': False, 'future_endpoint_label_eval_only': True, 'candidate_fde_input': False, 'candidate_fde_label_only': True, 'train_only_floor_selection': True, 'train_only_normalization_statistics': True, 'central_velocity': False, 'test_endpoint_goals': False, 'stage5c_executed': False, 'smc_enabled': False, 'validation_policy_selection_only': True, 'test_threshold_tuning': False, 'mixed_external_neural_proposal_used': False, 'mixed_external_floor_used': False, 'bootstrap_uses_test_labels_for_ci_only': True}`
