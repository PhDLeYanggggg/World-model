# Stage43-AL Bounded Residual Safety Audit

- source: `fresh_stage43_al_bounded_residual_safety_audit`
- result_source: `fresh_bounded_residual_audit_over_frozen_stage43_m_checkpoint`
- gate: `12 / 12`
- verdict: `stage43_al_bounded_residual_candidate_pass`
- deploy bounded residual: `True`
- stored policy replay max abs diff: `0.00000000`

## Policy Comparison

| policy | all | t50 | t100 diag | hard/failure | easy deg | switch | mean residual | safe easy | safe t100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| floor_only | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.0000 | `True` | `True` |
| ungated_neural_waypoint | 16.29% | 16.25% | -72.12% | 18.13% | 55.72% | 100.00% | 0.2577 | `False` | `False` |
| stored_stage43_m_hard_switch | 29.77% | 16.45% | -17.79% | 28.75% | 0.00% | 68.91% | 0.0000 | `True` | `False` |
| bounded_residual_unconstrained_val_best | 38.00% | 26.96% | 0.00% | 37.71% | 0.00% | 63.08% | 0.1320 | `True` | `True` |
| bounded_residual_safe_val_best | 38.00% | 26.96% | 0.00% | 37.71% | 0.00% | 63.08% | 0.1320 | `True` | `True` |

## Safe Bounded Residual vs Stored Hard Switch

- all delta: `8.23%`
- t50 delta: `10.52%`
- t100 diagnostic delta: `17.79%`
- hard/failure delta: `8.96%`
- easy degradation delta: `0.00%`

## Interpretation

- deployment decision: `promote_bounded_residual_candidate`
- Bounded residual is evaluated as a floor-protected relaxation of hard switching.
- If it is not better than the stored hard switch under validation-selected safety constraints, Stage43-M remains the active floor policy.
- Global floor removal is not supported.

## Boundary

- Dataset-local/raw-frame 2.5D only.
- Future waypoint/endpoint labels are loss/eval only.
- No metric/seconds claim, no Stage5C, no SMC.

## Gate

| gate | passed |
| --- | --- |
| stage43_m_exact_replay | `True` |
| feature_schema_and_rows_match | `True` |
| bounded_residual_search_completed | `True` |
| thresholds_selected_on_validation_only | `True` |
| safe_bounded_residual_preserves_easy | `True` |
| safe_bounded_residual_preserves_t100 | `True` |
| unsafe_or_unconstrained_risk_reported | `True` |
| deployment_decision_recorded | `True` |
| bounded_residual_lift_or_honest_diagnostic | `True` |
| global_floor_not_removed | `True` |
| no_future_or_test_leakage | `True` |
| no_metric_seconds_stage5c_smc_claim | `True` |
