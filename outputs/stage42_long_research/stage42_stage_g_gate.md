# Stage42-G Gate

- source: `fresh_run`
- passed: `11 / 11`
- verdict: `stage42_g_retrained_ablation_phase1_pass`

| gate | pass |
| --- | --- |
| `fresh_retrained_rows_present` | `True` |
| `three_seeds_per_fresh_variant` | `True` |
| `full_variant_safe` | `True` |
| `at_least_two_positive_component_contributions` | `True` |
| `no_safe_switch_diagnosed` | `True` |
| `no_teacher_floor_proxy_diagnosed` | `True` |
| `source_labels_explicit` | `True` |
| `no_leakage_pass` | `True` |
| `no_metric_seconds_overclaim` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |

- positive component contributions: `['no_domain_expert', 'no_goal', 'no_interaction', 'no_neighbor', 'no_safe_switch', 'no_scene_goal', 'no_teacher_floor_proxy']`
