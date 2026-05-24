# Stage35 Gates

- gates passed: `12 / 14`
- verdict: `stage35_external_selective_transfer_not_deployable`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass | evidence |
| --- | --- | --- |
| Gate1 external data expansion attempted | True | 18 |
| Gate2 external t50 rows enough or blocker | True | {'test_horizon_counts': {'10': 21267, '25': 18765, '50': 16263, '100': 10008}, 'blocker': None} |
| Gate3 external held-out scene split built | True | ['UCY_crowds', 'UCY_students01', 'UCY_zara03'] |
| Gate4 no leakage pass | True | {'future_endpoint_input': False, 'central_velocity': False, 'test_endpoint_goals': False} |
| Gate5 external hard/easy/failure labels built | True | {'train': {'rows': 158942, 'easy': 39736, 'hard': 131433, 'failure': 64528, 'oracle_headroom': 0.46283990144729614}, 'val': {'rows': 112746, 'easy': 35967, 'hard': 78283, 'failure': 39068, 'oracle_headroom': 0.5199509859085083}, 'test': {'rows': 66303, 'easy': 20798, 'hard': 45917, 'failure': 22891, 'oracle_headroom': 0.5287383794784546}} |
| Gate6 selective transfer all improvement > 0 | True | {'rows': 66303, 'all_improvement': 0.12131890857784355, 't10_improvement': 0.30620476404756647, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1398494448930071, 'easy_degradation': 0.0004114683717719725, 'selector_regret': 0.4289430569144169, 'switch_rate': 0.04999773765892946, 'mean_confidence': 0.03720055893063545} |
| Gate7 t50 improvement > 3 | False | {'rows': 66303, 'all_improvement': 0.12131890857784355, 't10_improvement': 0.30620476404756647, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1398494448930071, 'easy_degradation': 0.0004114683717719725, 'selector_regret': 0.4289430569144169, 'switch_rate': 0.04999773765892946, 'mean_confidence': 0.03720055893063545} |
| Gate8 hard/failure improvement > 10 | True | {'rows': 66303, 'all_improvement': 0.12131890857784355, 't10_improvement': 0.30620476404756647, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1398494448930071, 'easy_degradation': 0.0004114683717719725, 'selector_regret': 0.4289430569144169, 'switch_rate': 0.04999773765892946, 'mean_confidence': 0.03720055893063545} |
| Gate9 easy degradation <= 2 | True | {'rows': 66303, 'all_improvement': 0.12131890857784355, 't10_improvement': 0.30620476404756647, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1398494448930071, 'easy_degradation': 0.0004114683717719725, 'selector_regret': 0.4289430569144169, 'switch_rate': 0.04999773765892946, 'mean_confidence': 0.03720055893063545} |
| Gate10 SDD performance not destroyed | True | {'rows': 100000, 'all_improvement': 0.0, 't10_improvement': 0.0, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'selector_regret': 0.0, 'switch_rate': 0.0, 'mean_confidence': 0.0, 'status': 'cached_verified', 'reason': 'Mixed-domain deployment to SDD is disabled unless easy preservation is proven.'} |
| Gate11 held-out external scenes positive or blocker | True | {'rows': 66303, 'all_improvement': 0.12131890857784355, 't10_improvement': 0.30620476404756647, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1398494448930071, 'easy_degradation': 0.0004114683717719725, 'selector_regret': 0.4289430569144169, 'switch_rate': 0.04999773765892946, 'mean_confidence': 0.03720055893063545, 'status': 'fresh_run', 'reason': 'Held-out external test scenes under split v2.'} |
| Gate12 world model cross-domain candidate gate | False | False |
| Gate13 Stage5C false | True | Stage5C not executed |
| Gate14 SMC false | True | SMC not enabled |
