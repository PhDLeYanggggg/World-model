# Stage41 Route/Physical-Aware Policy Integration

- source: `fresh_run`
- best mode: `no_route_physical`
- route/physical contributes: `False`
- best metrics: `{'rows': 55528, 'all_improvement': 0.18577852429834418, 't10_improvement': 0.2639424206993144, 't25_improvement': 0.01350273464284224, 't50_improvement': 0.14803699577731477, 't100_improvement': 0.22857426649949408, 'hard_failure_improvement': 0.19518047277951456, 'easy_degradation': 0.0, 'harm_over_fallback': -0.08891914605641113, 'switch_rate': 0.2946441434951736, 'regret_to_oracle': -0.11002850459671851, 'by_domain': {'ETH_UCY': {'rows': 25901, 'all_improvement': 0.21976483138872083, 't50_improvement': 0.16111929638455114, 't100_improvement': 0.23250265575395312, 'hard_failure_improvement': 0.22611959337799148, 'easy_degradation': 0.0, 'switch_rate': 0.38554495965406743}, 'TrajNet': {'rows': 20087, 'all_improvement': 0.2472862268599434, 't50_improvement': 0.21884068163302184, 't100_improvement': 0.3718979141821237, 'hard_failure_improvement': 0.26577561826854845, 'easy_degradation': 0.0, 'switch_rate': 0.31736944292328373}, 'UCY': {'rows': 9540, 'all_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'switch_rate': 0.0}}, 'selected_source_distribution': {'0': 39167, '1': 16361}, 'route_switch_rate': {'stop': 0.1599300558208353, 'straight': 0.39209863791879157, 'left_turn': 0.1064198958935801, 'right_turn': 0.18910126067507116, 'reverse_or_uturn': 0.055690809494826535, 'interaction_detour': 0.45528824330458467}, 'all_ci': {'low': 0.1817630182582464, 'mid': 0.18578375329654784, 'high': 0.18973148050592917, 'n': 55528}, 't50_ci': {'low': 0.1415823368954965, 'mid': 0.14797498556112737, 'high': 0.15467506365352926, 'n': 13689}, 'hard_failure_ci': {'low': 0.19097044546882866, 'mid': 0.19516708658843623, 'high': 0.19915787315579425, 'n': 41741}}`
- lift over no-route/physical: `{'all_delta': 0.0, 't50_delta': 0.0, 't100_delta': 0.0, 'hard_delta': 0.0, 'easy_delta': 0.0}`
- lift over full trajectory reference: `{'all_delta': 0.0, 't50_delta': 0.0, 't100_delta': 0.0, 'hard_delta': 0.0, 'easy_delta': 0.0}`

| ablation | all | t50 | t100 | hard | easy | switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_route_physical | 0.185779 | 0.148037 | 0.228574 | 0.195180 | 0.000000 | 0.294644 |
| physical_only | 0.185779 | 0.148037 | 0.228574 | 0.195180 | 0.000000 | 0.294644 |
| route_only | 0.185858 | 0.147037 | 0.228372 | 0.195279 | 0.000000 | 0.295995 |
| route_physical | 0.185858 | 0.147037 | 0.228372 | 0.195279 | 0.000000 | 0.295995 |

- no leakage: `{'route_physical_predictions_from_past_only_models': True, 'future_route_label_input': False, 'future_physical_label_input': False, 'future_waypoints_label_eval_only': True, 'central_velocity': False, 'test_endpoint_goals': False}`
