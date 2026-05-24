# Stage32 Gates

- gates passed: `9 / 11`
- verdict: `stage32_domain_alignment_partial_not_cross_domain_candidate`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass | evidence |
| --- | --- | --- |
| Gate1 external reaudit pass | True | rows={'train': 119109, 'val': 7685, 'test': 3636} |
| Gate2 normalization built | True | ['raw_dataset_local', 'per_scene_zscore', 'velocity_scale', 'path_length_speed', 'robust_quantile'] |
| Gate3 external baseline recomputed | True | baseline by normalization exists |
| Gate4 latent alignment measured | True | mean_distance=6.645000520842287 |
| Gate5 adapted selector improves external strongest baseline or reduces domain gap | True | {'domain': 'external', 'split': 'test', 'rows': 3636, 'all_improvement': 0.0, 't10_improvement': 0.0, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'selector_regret': 0.01730417331328272, 'harm_over_fallback': 0.0, 'switch_rate': 0.0, 'mean_confidence': 0.0} |
| Gate6 mixed-domain training does not destroy SDD performance | False | {'domain': 'sdd', 'split': 'test', 'rows': 100000, 'all_improvement': 0.04593392521544648, 't10_improvement': 0.012675610213318533, 't25_improvement': 0.01660725927948936, 't50_improvement': 0.04481255195163136, 't100_improvement': 0.05667802793817034, 'hard_failure_improvement': 0.04629964467328851, 'easy_degradation': 250699.06445684892, 'selector_regret': 5.651677536930442, 'harm_over_fallback': -1.191947324232161, 'switch_rate': 0.05, 'mean_confidence': 0.038551878184080124} |
| Gate7 no leakage pass | True | {'split_by_file': True, 'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'candidate_goals_used': False} |
| Gate8 external generalization positive or honest blocker | True | not_cross_domain_candidate |
| Gate9 world model cross-domain candidate gate | False | best=0.0, mixed_ext=0.0, mixed_sdd=0.04593392521544648 |
| Gate10 Stage5C false plan only | True | Stage5C not executed |
| Gate11 SMC false | True | SMC not enabled |
