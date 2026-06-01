# Stage43-AP Paper Evidence Refresh

- source: `fresh_stage43_ap_paper_evidence_refresh`
- result_source: `fresh_paper_evidence_refresh_from_stage43_aj_to_ao_plus_stage43_p`
- gate: `10 / 10`
- verdict: `stage43_ap_paper_evidence_refresh_pass`
- policy hash: `03497313f878a1ec69fd7d2824842fee0acfa79c38dc9d667c6d6ac53ef4c331`
- frozen replayable policy hash: `4dc482d146af2940b4968385e3f4bdf6951036b27b0353f1759a2130265ed493`

## Key Current Metrics

- all: `50.25%`
- endpoint: `51.15%`
- t50: `51.23%`
- t50 endpoint: `55.13%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure: `47.88%`
- easy degradation: `0.00%`
- switch rate: `70.45%`
- latest t50 CI: `[50.76%, 51.74%]`
- latest all CI: `[49.96%, 50.53%]`
- reviewer replay diff: `0.00000000`

## Frozen Replayable Safety Artifact

- frozen all: `38.00%`
- frozen t50: `26.96%`
- frozen hard/failure: `37.71%`
- frozen easy: `0.00%`
- latest-vs-frozen all delta: `12.25%`
- latest-vs-frozen t50 delta: `24.27%`
- latest-vs-frozen hard delta: `10.16%`

## Claim Evidence Table

| claim | status | allowed claim | caveat |
| --- | --- | --- | --- |
| Reviewer-replayable protected bounded residual policy | `supported` | Stage43 bounded residual policy is frozen, hashable, and exact-replayable from the policy artifact. | Requires local checkpoint/cache not committed to git; dataset-local/raw-frame only. |
| Latest protected tail-horizon full-waypoint adapter | `supported` | Stage43-P is the current strongest protected full-waypoint evidence block under the safety floor. | It is validation-selected and floor-protected; h100/t100 is fallback-guarded and not solved. |
| Protected full-waypoint latent dynamics lift | `supported` | Protected full-waypoint dynamics improves the frozen replayable policy while preserving easy cases. | This is protected dynamics, not ungated generative rollout and not global floor removal. |
| Bootstrap-supported latest full-test lift | `supported` | Stage43-P has bootstrap-positive all/t50/hard full-test evidence with zero easy degradation. | Bootstrap is over dataset-local/raw-frame rows; not metric/seconds evidence. |
| Frozen replay bootstrap-supported delta over stored hard switch | `supported` | The frozen bounded-residual safety artifact has positive bootstrap delta over stored Stage43-M hard switch on all/t50/hard slices. | This remains the exact replayable safety artifact; Stage43-P is stronger but not yet frozen as the primary reviewer replay artifact. |
| Per-domain external support | `partially_supported` | Stage43-P is positive on ETH_UCY and UCY and safely floors TrajNet to non-harmful zero transfer. | This is not uniform positive transfer across every source/domain; metric/seconds calibration remains unverified. |
| Global floor removal | `not_supported` | Do not remove the safety floor globally. | The floor is currently part of the method, not a disposable crutch. |
| h100 / t100 raw-frame behavior | `guarded_only` | Report t100 only as raw-frame diagnostic with h100 floor guard. | No seconds-level long-horizon claim. |
| A-journal readiness | `not_yet` | The current package is a stronger candidate evidence block, not a complete A-journal submission claim. | Need source-level calibration, more multimodal scene evidence, full paper package coherence, and broader external verification. |

## Direct Answers

- still 2.5D: `True`
- metric/time subset available: `False`
- full-waypoint dynamics available: `True`
- cross-domain external evidence: Stage43-P is positive on ETH_UCY and UCY and safely floors TrajNet to zero/non-harm; uniform positive per-source transfer remains blocked.
- A-journal candidate now: `False`

## Gate

| gate | passed |
| --- | --- |
| replayable_policy_claim_supported | `True` |
| latest_tail_adapter_claim_supported | `True` |
| full_waypoint_claim_supported | `True` |
| latest_bootstrap_claim_supported | `True` |
| frozen_replay_bootstrap_claim_supported | `True` |
| global_floor_not_overclaimed | `True` |
| a_journal_not_overclaimed | `True` |
| claim_boundary_answers_present | `True` |
| no_future_or_test_leakage | `True` |
| no_metric_seconds_stage5c_smc_claim | `True` |
