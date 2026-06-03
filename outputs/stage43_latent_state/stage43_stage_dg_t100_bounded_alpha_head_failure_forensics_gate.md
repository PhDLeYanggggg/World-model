# Stage43-DG Gate

- verdict: `stage43_dg_t100_bounded_alpha_head_forensics_selection_gap_identified`
- passed: `13 / 13`
- failure root: `validation_group_risk_selection_gap`
- selection misses safe candidate: `True`
- deploy on current heldout t100: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass |
| --- | --- |
| `df_precondition_present` | `True` |
| `fresh_failure_forensics` | `True` |
| `three_seed_replay` | `True` |
| `replay_diff_zero` | `True` |
| `candidate_search_completed` | `True` |
| `positive_candidate_availability_measured` | `True` |
| `selection_gap_measured` | `True` |
| `root_cause_identified` | `True` |
| `diagnostic_not_deployed` | `True` |
| `test_oracle_marked_diagnostic_only` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
