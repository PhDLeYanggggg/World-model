# Stage42-IF Gate

- source: `fresh_stage42_if_t50_gain_harm_stability_audit`
- passed: `13 / 14`
- verdict: `stage42_if_t50_gain_harm_ci_blocker_identified`

| gate | pass |
| --- | --- |
| `stage42p_artifact_loaded` | `True` |
| `stability_audit_completed` | `True` |
| `mean_ade_t50_positive` | `True` |
| `paper_stable_ade_t50_ci_positive` | `False` |
| `fde_t50_ci_positive` | `True` |
| `negative_seed_identified` | `True` |
| `domain_instability_identified` | `True` |
| `validation_selected_seed_positive` | `True` |
| `row_bootstrap_availability_audited` | `True` |
| `no_future_endpoint_or_waypoint_input` | `True` |
| `no_central_velocity_or_test_goal` | `True` |
| `no_metric_seconds_overclaim` | `True` |
| `stage5c_false` | `True` |
| `smc_false` | `True` |
