# Stage38 Bounded Correction Report

- source: `fresh_run`
- correction form: `prediction = selected_baseline + alpha * bounded_delta`

| variant | all | t50 | hard | easy | intervention |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear_bounded_correction | 0.134825 | 0.084573 | 0.155434 | 0.000411 | 0.048912 |
| ridge_correction | 0.134825 | 0.084573 | 0.155434 | 0.000411 | 0.048912 |
| horizon_specific_correction | 0.134825 | 0.084573 | 0.155434 | 0.000411 | 0.048912 |
| hard_only_correction | 0.134825 | 0.084573 | 0.155434 | 0.000411 | 0.048912 |
| t50_only_correction | 0.134825 | 0.084573 | 0.155434 | 0.000411 | 0.048912 |
| small_mlp_correction | not_run | not_run | not_run | not_run | not_run |

- best variant: `linear_bounded_correction`
- best metrics: `{'rows': 66303, 'all_improvement': 0.1348254070040389, 't10_improvement': 0.30620476404756647, 't25_improvement': 0.0, 't50_improvement': 0.08457292499203606, 't100_improvement': 0.0, 'hard_failure_improvement': 0.15543403861117056, 'easy_degradation': 0.0004114683717719725, 'harm_over_fallback': -0.14194807770788026, 'intervention_rate': 0.04891181394507036, 'smoothness_proxy': 1.859190279676184, 'physical_validity_proxy': 'bounded_delta_clip'}`
