# Stage43-AK Gate

- verdict: `stage43_ak_self_gate_conformal_audit_pass`
- passed: `12 / 12`
- global_floor_removable: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | passed |
| --- | --- |
| stage43_m_checkpoint_present | `True` |
| feature_schema_matches_checkpoint | `True` |
| cache_row_hashes_match_prior | `True` |
| stored_policy_exact_replay | `True` |
| ungated_neural_reported_unsafe | `True` |
| fresh_self_gate_eval_completed | `True` |
| conformal_style_gate_eval_completed | `True` |
| conformal_style_h100_guard_safe | `True` |
| self_gate_preserves_easy_on_at_least_one_policy | `True` |
| global_floor_still_required_if_ungated_unsafe | `True` |
| no_future_or_test_leakage | `True` |
| no_metric_seconds_stage5c_smc_claim | `True` |
