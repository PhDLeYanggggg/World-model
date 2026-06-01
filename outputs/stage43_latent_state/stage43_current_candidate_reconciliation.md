# Stage43-AY Current Candidate Reconciliation

- source: `fresh_stage43_ay_current_candidate_reconciliation`
- result_source: `fresh_reconciliation_from_stage43_p_ap_ao_ax_aq`
- verdict: `stage43_ay_current_candidate_reconciliation_pass`
- gate: `12 / 12`
- current candidate supported: `True`
- long objective complete: `False`

## Why This Exists

Stage43 now has several valid evidence artifacts with different roles. This reconciliation keeps the roles separate instead of pretending there is one universal winner.

## Role Map

| role | artifact | all | t50 | t100 raw | hard/failure | easy | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| performance leader | Stage43-P | `50.25%` | `51.23%` | `0.00%` | `47.88%` | `0.00%` | strongest aggregate protected full-waypoint evidence |
| source-horizon replay leader | Stage43-AX | `23.40%` | `12.80%` | `1.35%` | `24.73%` | `0.00%` | exact replay and all sources nonnegative on all-test |
| frozen reviewer artifact | Stage43-AO | `38.00%` | `26.96%` | `0.00%` | `37.71%` | `0.00%` | frozen bounded-residual reviewer replay |

## Source Boundary

- Stage43-P positive all-test domains: `2 / 3`
- Stage43-P nonnegative all-test domains: `3 / 3`
- Stage43-P uniform positive transfer claim allowed: `False`
- Stage43-AX nonnegative all-test domains: `3 / 3`

## Current Public Claim

M3W currently has protected dataset-local/raw-frame latent/full-waypoint evidence; Stage43-P is the performance leader, Stage43-AX is the source/horizon replay leader, and Stage43-AO remains the frozen reviewer-replayable artifact.

## Do Not Say

- Do not call this true 3D.
- Do not call this foundation-scale.
- Do not call dataset-local/raw-frame horizons metric or seconds-level.
- Do not claim uniform positive transfer across every source from Stage43-P.
- Do not say Stage5C or SMC has run.

## Next Required Evidence

- Freeze and exact-replay the Stage43-P performance leader if it is to replace Stage43-AO as the primary reviewer artifact.
- Continue source-level repair for TrajNet non-floor positive transfer under the performance leader.
- Refresh full-suite replay and paper tables from the reconciled role map.
- Source-specific timing/geometry calibration is still required before metric or seconds-level language.

## Gate

| gate | passed |
| --- | --- |
| `input_gate_verdicts_present` | `True` |
| `performance_leader_supported` | `True` |
| `source_horizon_replay_supported` | `True` |
| `frozen_reviewer_artifact_supported` | `True` |
| `roles_not_collapsed` | `True` |
| `uniform_positive_transfer_not_overclaimed` | `True` |
| `source_safe_candidate_has_all_sources_nonnegative` | `True` |
| `t100_raw_frame_guarded` | `True` |
| `long_objective_kept_active` | `True` |
| `no_future_or_test_leakage` | `True` |
| `claim_boundary_not_overstated` | `True` |
| `stage5c_and_smc_false` | `True` |
