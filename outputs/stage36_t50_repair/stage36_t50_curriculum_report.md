# Stage36 t+50 Curriculum Adaptation

- source: `fresh_run`

| adaptation | status | test t50 | all | hard | easy | switch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| t50_oversampling | fresh_run | -0.011639 | -0.001859 | -0.002145 | 0.000000 | 0.049048 |
| t50_hard_failure_oversampling | fresh_run | -0.011639 | -0.001859 | -0.002145 | 0.000000 | 0.049048 |
| t50_only_selector_refit | fresh_run | -0.006854 | -0.001095 | -0.001263 | 0.000000 | 0.029953 |
| horizon_balanced_loss | fresh_run | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| long_track_only_training | fresh_run | 0.000029 | 0.000005 | 0.000005 | 0.000000 | 0.049048 |
| per_scene_t50_selector | not_run | not_run | not_run | not_run | not_run | not_run |
| pedestrian_only_t50_selector | not_run | not_run | not_run | not_run | not_run | not_run |
| t50_goal_aware_selector | fresh_run | not_run | not_run | not_run | not_run | not_run |

- best adaptation: `long_track_only_training`
- best metrics: `{'rows': 66303, 'all_improvement': 4.552759222953284e-06, 't10_improvement': 0.0, 't25_improvement': 0.0, 't50_improvement': 2.850777100926738e-05, 't100_improvement': 0.0, 'hard_failure_improvement': 5.253241850633472e-06, 'easy_degradation': 0.0, 'selector_regret': 0.5566663101904641, 'harm_over_fallback': -4.7932762402102875e-06, 'switch_rate': 0.049047554409302745, 'mean_confidence': 0.010579731315374374}`
- t50 gate passed: `False`
