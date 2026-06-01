# Stage43 Current World-Model Gate

- source: `fresh_stage43_az_tail_adapter_reviewer_replay`
- verdict: `stage43_az_tail_adapter_reviewer_replay_pass`
- passed: `12 / 12`
- current performance leader replayed: `True`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

## Candidate Roles

- Performance leader and replayed candidate: `Stage43-P / Stage43-AZ tail adapter replay`.
- Source-horizon replay leader remains: `Stage43-AX exact replay of source-horizon expert policy`.
- Frozen bounded-residual artifact remains: `Stage43-AO`.

| gate | passed |
| --- | --- |
| stage43_p_artifact_present | `True` |
| stage43_p_artifact_passed | `True` |
| model_hash_exact | `True` |
| feature_standardization_hashes_match | `True` |
| split_hashes_recorded | `True` |
| switch_hash_recorded | `True` |
| replay_metrics_exact | `True` |
| replayed_policy_safe | `True` |
| no_future_or_test_leakage | `True` |
| claim_boundary_not_overstated | `True` |
| stage5c_and_smc_false | `True` |
| long_objective_kept_active | `True` |
