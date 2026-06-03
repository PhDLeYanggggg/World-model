# Stage43-DE Gate

- verdict: `stage43_de_t100_alpha_stability_policy_repairs_group_fragility_diagnostic`
- passed: `15 / 15`
- all bounded min-without-group positive: `True`
- repairs DD seed fragility: `True`
- beats CZ t100 mean: `True`
- deploy on current heldout t100: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass |
| --- | --- |
| `stage43_dd_precondition_present` | `True` |
| `fresh_alpha_stability_policy` | `True` |
| `three_seed_replay` | `True` |
| `replay_diff_zero` | `True` |
| `validation_only_policy_selection` | `True` |
| `bounded_alpha_protocol_used` | `True` |
| `safe_bounded_candidates_found` | `True` |
| `easy_preserved` | `True` |
| `t100_positive_all_seeds` | `True` |
| `all_min_without_group_positive` | `True` |
| `repairs_dd_seed_fragility` | `True` |
| `diagnostic_not_deployed` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
