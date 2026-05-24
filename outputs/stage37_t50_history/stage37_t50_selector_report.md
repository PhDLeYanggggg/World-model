# Stage37 t+50 Selector Report

- source: `fresh_run`

| selector | val t50 | test t50 | all | hard | easy | harm | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| history_only_t50_selector | 0.069811 | 0.083891 | 0.013398 | 0.015459 | 0.000000 | -0.014105 | 0.049048 |
| prototype_goal_t50_selector | 0.050810 | 0.066797 | 0.010668 | 0.012309 | 0.000000 | -0.011231 | 0.049048 |
| history_plus_goal_t50_selector | 0.050810 | 0.066797 | 0.010668 | 0.012309 | 0.000000 | -0.011231 | 0.049048 |
| neighbor_history_t50_selector | 0.070364 | 0.084573 | 0.013506 | 0.015585 | 0.000000 | -0.014220 | 0.049048 |
| mixture_of_experts_t50_selector | 0.069811 | 0.083891 | 0.013398 | 0.015459 | 0.000000 | -0.014105 | 0.049048 |
| conformal_safe_t50_selector | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

- best selector: `neighbor_history_t50_selector`
- best metrics: `{'rows': 66303, 'all_improvement': 0.01350649849487695, 't10_improvement': 0.0, 't25_improvement': 0.0, 't50_improvement': 0.08457292542209705, 't100_improvement': 0.0, 'hard_failure_improvement': 0.015584593797412505, 'easy_degradation': 0.0, 'selector_regret': 0.6213934391589183, 'harm_over_fallback': -0.01422003122790308, 'switch_rate': 0.049047554409302745, 'mean_confidence': 0.04121534153819084}`
- deployable: `False`
