# Stage42 Failure Taxonomy

## Confirmed Failures / Negative Evidence

- Ungated endpoint/full-waypoint neural is not deployable: easy degradation remains unsafe.
- Internal self-gate, uncertainty gate, harm gate, and conformal-risk gate can produce large raw lift but violate proximity/collision safety in the fresh Stage42-E study.
- JEPA-only and JEPA+Transformer hybrid attempts remain negative or fallback-only in cached-verified same-protocol architecture evidence.
- Full Stage42-D retraining of every named component has not been completed; Stage42-D is an evidence audit with fresh safety/waypoint rows plus cached-verified Stage30/41 component evidence.
- Metric/time claims remain blocked by missing verified homography/FPS/stride calibration for the pedestrian external benchmark.

## Root Causes

- Safety/floor dependence: neural proposals can help hard/long-horizon slices but are too risky without the teacher floor.
- Calibration gap: dataset-local and raw-frame coordinates prevent metric/seconds claims.
- Evidence gap: some contribution claims rely on cached-verified prior ablations rather than fresh Stage42 retraining.
- Data gap: broader legally verified top-down external domains remain needed for stronger generalization claims.

## Current Best Safe Action

Keep `current_composite_tail_policy` as the deployable policy. Do not execute Stage5C or SMC.

<!-- STAGE42_AC_REFRESH:START -->
## Stage42-AC Latest Evidence Refresh

- source: `fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-AB is now included as auxiliary-head evidence.
- Auxiliary-head conclusion: mixed / partial; not a uniform main contribution claim.
- Stage5C and SMC remain disabled.

### Newly Confirmed Failure / Limitation

- Auxiliary interaction/occupancy/physical heads are not uniformly positive: Stage42-AB shows small t50/FDE@50 support but negative all/hard ADE deltas. Treat them as partial/mixed evidence, not as a main contribution.
- Ungated neural replacement, metric/seconds-level claims, true 3D, foundation claims, Stage5C, and SMC remain rejected/not enabled.
<!-- STAGE42_AC_REFRESH:END -->

<!-- STAGE42_AJ_REFRESH:START -->
## Stage42-AJ Post-Repair Paper Package Refresh

- source: `fresh_synthesis_from_stage42_ad_to_ai_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- This refresh supersedes stale Stage42-AE limitations: horizon=25 harm, ETH_UCY t50/FDE, and TrajNet|100 easy safety were repaired by validation-only guards.
- t100 remains raw-frame diagnostic; metric/seconds/true-3D/foundation/Stage5C/SMC claims remain rejected.
- Future waypoints/endpoints remain labels/eval only, never inference inputs.

### Post-Repair Headline Metrics

- ADE all CI low: `0.085978`
- ADE t50 CI low: `0.058513`
- ADE t100 raw-frame diagnostic CI low: `0.068349`
- ADE hard/failure CI low: `0.090662`
- easy degradation CI high: `0.001168`
- FDE@50 CI low: `0.148230`

### Evidence Rows

| item | status | paper use | evidence |
| --- | --- | --- | --- |
| Stage42-AD calibration evidence refresh | `stage42_ad_calibration_evidence_refresh_pass` | data/calibration boundary | audited=7, files=1152, metric_allowed=False, seconds_allowed=False |
| Stage42-AF horizon25 validation-margin guard | `stage42_af_weak_slice_guard_repair_pass_with_eth_t50_limitation` | weak-slice safety repair | horizon25 -0.004781149088858072 -> 0.0; validation-only low-margin guard |
| Stage42-AG ETH_UCY t50/FDE source repair | `stage42_ag_eth_t50_fde_source_repair_pass` | domain t50/FDE lower-bound repair | ADE@50 low -0.013218100958604987 -> 0.002820688160982139; FDE@50 low -0.04199023614248535 -> 0.021040393452369632 |
| Stage42-AH post-repair claim matrix | `stage42_ah_post_repair_claim_refresh_pass` | claim matrix and remaining limitations | all_low=0.085258, t50_low=0.058513, hard_low=0.089767, easy_high=0.003348 |
| Stage42-AI TrajNet t100 easy-safety repair | `stage42_ai_trajnet_t100_safety_repair_pass` | raw-frame diagnostic t100 safety repair | TrajNet100 easy high 0.08498424090178214 -> 0.0; global t100 raw-frame low=0.068349 |

### Claim Boundary

- Supported: protected row-level full-waypoint raw-frame 2.5D world-state evidence with positive all/t50/hard/FDE@50 lower bounds and repaired t100 easy-safety diagnostic.
- Supported as non-harm only: horizon=25 floor/non-harm slices; do not call them positive dynamics contributions.
- Rejected: metric prediction, seconds-level horizon, true 3D, foundation model, Stage5C execution, SMC readiness, and ungated neural deployment.
<!-- STAGE42_AJ_REFRESH:END -->
