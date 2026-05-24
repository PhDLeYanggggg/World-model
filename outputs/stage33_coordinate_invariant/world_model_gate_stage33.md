# Stage33 Gates

- gates passed: `11 / 13`
- verdict: `stage33_coordinate_invariant_partial_not_cross_domain_candidate`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass | evidence |
| --- | --- | --- |
| Gate1 external scene/goal context built or blocker | True | 2 |
| Gate2 coordinate-invariant feature schema built | True | 8e24edb1e76194a3fe72ecca85b0fad7970a7d68653a20b337a79f3d2d27323c |
| Gate3 relative baselines recomputed | True | relative_baseline_metrics.json |
| Gate4 domain adapter reduces latent gap | True | 0.9999999999999976 |
| Gate5 external selector improves external strongest baseline | False | {'domain': 'external', 'split': 'test', 'rows': 3636, 'all_improvement': 0.0, 't10_improvement': 0.0, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'selector_regret': 0.01730417331328272, 'harm_over_fallback': 0.0, 'switch_rate': 0.0, 'mean_confidence': 0.0} |
| Gate6 mixed/domain-conditioned model preserves SDD easy <=2 | True | {'domain': 'sdd', 'split': 'test', 'rows': 100000, 'all_improvement': 0.0, 't10_improvement': 0.0, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'selector_regret': 6.843624861162603, 'harm_over_fallback': 0.0, 'switch_rate': 0.0, 'mean_confidence': 0.0} |
| Gate7 SDD performance not destroyed | True | {'domain': 'sdd', 'split': 'test', 'rows': 100000, 'all_improvement': 0.0, 't10_improvement': 0.0, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'selector_regret': 6.843624861162603, 'harm_over_fallback': 0.0, 'switch_rate': 0.0, 'mean_confidence': 0.0} |
| Gate8 external transfer positive or honest blocker | True | not_cross_domain_candidate |
| Gate9 cross-domain world-model candidate gate | False | best=0.0, mixed_ext=0.0, mixed_sdd_easy=0.0 |
| Gate10 no leakage pass | True | {'features': {'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False, 'test_statistics_normalization': False}, 'scene': {'candidate_goals_from_train_only': True, 'test_endpoints_used': False, 'future_endpoint_input': False, 'central_velocity': False}} |
| Gate11 no metric/seconds overclaim | True | dataset-local/pixel raw-frame only |
| Gate12 Stage5C false plan only | True | Stage5C not executed |
| Gate13 SMC false | True | SMC not enabled |
