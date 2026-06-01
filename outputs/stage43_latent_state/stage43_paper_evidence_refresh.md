# Stage43-AP Paper Evidence Refresh

- source: `fresh_stage43_ap_paper_evidence_refresh`
- result_source: `fresh_paper_evidence_refresh_from_stage43_aj_to_ao`
- gate: `8 / 8`
- verdict: `stage43_ap_paper_evidence_refresh_pass`
- policy hash: `4dc482d146af2940b4968385e3f4bdf6951036b27b0353f1759a2130265ed493`

## Key Current Metrics

- all: `38.00%`
- t50: `26.96%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure: `37.71%`
- easy degradation: `0.00%`
- t50 delta CI vs stored hard switch: `[9.86%, 11.21%]`
- reviewer replay diff: `0.00000000`

## Claim Evidence Table

| claim | status | allowed claim | caveat |
| --- | --- | --- | --- |
| Reviewer-replayable protected bounded residual policy | `supported` | Stage43 bounded residual policy is frozen, hashable, and exact-replayable from the policy artifact. | Requires local checkpoint/cache not committed to git; dataset-local/raw-frame only. |
| Protected full-waypoint latent dynamics lift | `supported` | Protected bounded residual latent waypoint policy improves full-waypoint metrics under safety floor. | This is protected residual dynamics, not ungated generative rollout. |
| Bootstrap-supported delta over stored hard switch | `supported` | Bounded residual has positive bootstrap delta over stored Stage43-M hard switch on all/t50/hard slices. | Bootstrap over frozen rows; not a new external dataset acquisition claim. |
| Per-domain external support | `partially_supported` | Positive dataset-local/raw-frame deltas are observed across ETH_UCY, TrajNet, and UCY slices. | Domain labels are from existing external conversion; metric/seconds calibration remains unverified. |
| Global floor removal | `not_supported` | Do not remove the safety floor globally. | The floor is currently part of the method, not a disposable crutch. |
| h100 / t100 raw-frame behavior | `guarded_only` | Report t100 only as raw-frame diagnostic with h100 floor guard. | No seconds-level long-horizon claim. |
| A-journal readiness | `not_yet` | The current package is a stronger candidate evidence block, not a complete A-journal submission claim. | Need source-level calibration, more multimodal scene evidence, full paper package coherence, and broader external verification. |

## Direct Answers

- still 2.5D: `True`
- metric/time subset available: `False`
- full-waypoint dynamics available: `True`
- cross-domain external evidence: positive dataset-local/raw-frame slices for ETH_UCY, TrajNet, UCY in Stage43-AM
- A-journal candidate now: `False`

## Gate

| gate | passed |
| --- | --- |
| replayable_policy_claim_supported | `True` |
| full_waypoint_claim_supported | `True` |
| bootstrap_delta_claim_supported | `True` |
| global_floor_not_overclaimed | `True` |
| a_journal_not_overclaimed | `True` |
| claim_boundary_answers_present | `True` |
| no_future_or_test_leakage | `True` |
| no_metric_seconds_stage5c_smc_claim | `True` |
