# Stage43 Current World-Model Gate

- source: `fresh_stage43_ay_current_candidate_reconciliation`
- verdict: `stage43_ay_current_candidate_reconciliation_pass`
- passed: `12 / 12`
- current candidate supported: `True`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

## Candidate Roles

- Performance leader: `Stage43-P protected tail-horizon full-waypoint adapter`.
- Source-horizon replay leader: `Stage43-AX exact replay of source-horizon expert policy`.
- Frozen reviewer artifact: `Stage43-AO bounded-residual replay`.

| gate | passed |
| --- | --- |
| input_gate_verdicts_present | `True` |
| performance_leader_supported | `True` |
| source_horizon_replay_supported | `True` |
| frozen_reviewer_artifact_supported | `True` |
| roles_not_collapsed | `True` |
| uniform_positive_transfer_not_overclaimed | `True` |
| source_safe_candidate_has_all_sources_nonnegative | `True` |
| t100_raw_frame_guarded | `True` |
| long_objective_kept_active | `True` |
| no_future_or_test_leakage | `True` |
| claim_boundary_not_overstated | `True` |
| stage5c_and_smc_false | `True` |
