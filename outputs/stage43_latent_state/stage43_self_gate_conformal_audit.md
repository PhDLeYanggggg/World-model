# Stage43-AK Self-Gate / Conformal-Style Safety Audit

- source: `fresh_stage43_ak_self_gate_conformal_audit`
- result_source: `fresh_replay_and_audit_over_frozen_stage43_m_checkpoint`
- gate: `12 / 12`
- verdict: `stage43_ak_self_gate_conformal_audit_pass`
- stored policy replay max abs diff: `0.00000000`
- cache row hashes match prior: `True`
- feature schema match: `True`

## Policy Comparison

| policy | all | t50 | t100 diag | hard/failure | easy deg | switch | safe easy | safe t100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| floor_only | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | `True` | `True` |
| ungated_neural | 16.29% | 16.25% | -72.12% | 18.13% | 55.72% | 100.00% | `False` | `False` |
| stored_stage43_m_self_gate | 29.77% | 16.45% | -17.79% | 28.75% | 0.00% | 68.91% | `True` | `False` |
| fresh_self_gate_search | 29.77% | 16.45% | -17.79% | 28.75% | 0.00% | 68.91% | `True` | `False` |
| conformal_style_h100_easy_guard | 32.41% | 16.45% | 0.00% | 31.96% | 0.00% | 63.08% | `True` | `True` |

## Interpretation

- best safe policy by t50 in this audit: `stored_stage43_m_self_gate` with t50 `16.45%` and easy degradation `0.00%`.
- Ungated neural deployment remains unsafe and is not promoted.
- The conformal-style guard is validation-calibrated and explicitly floors h100; it is a diagnostic safety audit, not a formal conformal guarantee.
- Global safety floor removal is still not supported.

## Boundary

- Dataset-local/raw-frame 2.5D only.
- Future waypoints/endpoints are labels/eval only, not inputs.
- No metric/seconds claim, no Stage5C, no SMC.

## Gate

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
