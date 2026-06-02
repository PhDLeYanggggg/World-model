# Stage43-BU Context Admissibility Robustness Gate

- verdict: `stage43_bu_context_admissibility_partial_robust_lift_pass`
- passed: `12 / 12`
- robust all/hard lift: `True`
- t50 bootstrap robust: `True`
- t100 bootstrap robust: `False`
- t100 CI crosses zero: `True`
- slice easy safe: `False`
- easy-safe CI: `True`
- deployable policy changed: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | passed |
| --- | --- |
| `bt_precondition_passed` | `True` |
| `checkpoint_replayed_not_committed` | `True` |
| `exact_replay_matches_bt_report` | `True` |
| `bootstrap_completed` | `True` |
| `slice_audit_completed` | `True` |
| `slice_easy_hazards_reported` | `True` |
| `all_and_hard_bootstrap_measured` | `True` |
| `easy_safety_ci_measured` | `True` |
| `t50_and_t100_reported` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
