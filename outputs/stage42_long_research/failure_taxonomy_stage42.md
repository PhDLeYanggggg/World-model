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

<!-- STAGE42_CA_REFRESH:START -->
## Stage42-CA Post-BZ Paper Evidence Refresh

- source: `fresh_synthesis_from_stage42_by_bz_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-BY repaired the protected t50 slices; Stage42-BZ adds bootstrap evidence.
- This is protected policy evidence under the Stage37/teacher floor, not floor-free neural world dynamics.
- t+50/t+100 remain raw-frame horizons; no global metric or seconds-level claim is allowed.
- Stage5C remains unexecuted and SMC remains disabled.

### Post-BZ Headline Evidence

- selected variant: `family_baseline_rel_only`
- internal validation group: `UCY::UCY/zara03/crowds_zara03.txt`
- robust t50 slices: `TrajNet|50, UCY|50`
- target union t50 CI: `[28.52%, 29.45%]`
- target union hard/failure CI low: `28.51%`
- target union easy degradation CI high: `-25.16%`
- bootstrap_n: `3000`

### Evidence Rows

| item | status | paper use | evidence |
| --- | --- | --- | --- |
| Stage42-BY protected t50 floor-relaxability repair | `stage42_by_t50_floor_relaxability_repair_pass` | point-estimate protected t50 slice repair | repaired=TrajNet|50, UCY|50; global t50=28.97%; easy=-37.05%; not floor-free neural |
| Stage42-BZ protected t50 bootstrap evidence | `stage42_bz_t50_repair_statistical_evidence_pass` | bootstrap-backed t50 statistical evidence | target union t50 CI=[28.52%, 29.45%]; hard CI low=28.51%; easy CI high=-25.16%; n=3000 |
| Stage42-BZ slice TrajNet|50 | `ci_positive_and_easy_safe` | slice-level t50 evidence | rows=9198; t50 CI=[29.80%, 30.67%]; easy CI high=-27.61%; switch=95.26% |
| Stage42-BZ slice UCY|50 | `ci_positive_and_easy_safe` | slice-level t50 evidence | rows=2340; t50 CI=[23.02%, 26.08%]; easy CI high=-8.16%; switch=65.00% |

### Claim Boundary

- Supported: protected t50 slice repair with bootstrap evidence for `TrajNet|50` and `UCY|50`.
- Still required: teacher/floor rollout context, protected safe-switch, train/internal-validation policy selection.
- Rejected: true 3D, foundation model, global metric prediction, seconds-level horizon, Stage5C execution, SMC readiness, and ungated/floor-free neural deployment.
<!-- STAGE42_CA_REFRESH:END -->

<!-- STAGE42_CL_CONTEXT_GUARD_REFRESH:START -->
## Stage42-CL Post-CJ/CK Context Guard Refresh

- source: `fresh_synthesis_from_stage42_cj_ck_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-CJ tested whether goal/scene context can become a validation-only gated expert over baseline-family rollout context.
- Stage42-CK tested whether scalar neighbor or kNN interaction graph context can become a validation-only gated expert.
- Both gates selected `baseline_family_control`, so goal/scene and neighbor/interaction remain diagnostic/auxiliary rather than main claims.
- This refresh updates paper-package language to prevent context overclaims.
- Stage5C remains unexecuted and SMC remains disabled.

### Evidence Rows

| item | status | paper use | evidence |
| --- | --- | --- | --- |
| Stage42-CJ goal/scene gated expert | `diagnostic_negative` | claim boundary / limitation | gate=10/10; selected=baseline_family_control; goal_scene_rescue_success=False; control all/t50/hard=28.78%/31.54%/27.58%; goal all/t50/hard=26.25%/22.76%/24.86% |
| Stage42-CJ motion+goal context | `diagnostic_negative` | ablation boundary | motion_goal all/t50/hard=24.58%/22.02%/23.75%; delta_t50_vs_control=-9.53% |
| Stage42-CK neighbor/interaction gated expert | `diagnostic_negative` | claim boundary / limitation | gate=11/11; selected=baseline_family_control; neighbor_interaction_rescue_success=False; graph_rows=337991; rows_with_neighbors=334525; control all/t50/hard=28.78%/31.54%/27.58% |
| Stage42-CK graph/scalar candidates | `diagnostic_negative` | ablation boundary | scalar all/t50/hard=26.37%/22.96%/24.88%; knn_graph all/t50/hard=24.38%/22.38%/23.78%; graph_goal all/t50/hard=20.67%/22.21%/18.81% |

### Claim Boundary

- Supported main mechanism remains baseline-family rollout context plus causal history under a conservative safety floor.
- Goal/scene context is not a standalone main contribution under the current source-level ridge/full-waypoint protocol.
- Neighbor/interaction context is not a standalone main contribution under the current source-level ridge/full-waypoint protocol.
- Rejected: true 3D, foundation model, global metric prediction, seconds-level horizon, Stage5C execution, SMC readiness, and ungated/floor-free neural deployment.
<!-- STAGE42_CL_CONTEXT_GUARD_REFRESH:END -->

<!-- STAGE42_DR_POST_DQ_PAPER_REFRESH:START -->
## Stage42-DR Post-DP/DQ Paper Evidence Refresh

- source: `fresh_paper_refresh_from_stage42_dp_dq_dn_do`
- role: synchronize paper-ready evidence after the fresh context-closure and full-waypoint-promotion checkpoints.
- This is not new training and not a threshold search; it updates claim hygiene and paper artifacts.

### Context Claim Boundary

- closure decision: `close_current_sequence_graph_residual_context_protocol`.
- best context deltas vs baseline-family control all/t50/hard: `-2.30%` / `-8.31%` / `-2.62%`.
- positive context rows: `[]`.
- Paper wording: sequence/graph/neighbor/goal context remains auxiliary or diagnostic under the current residual protocol, not an independent main contribution.

### Full-Waypoint Runtime Evidence

- runtime all/t50/t100 raw/hard vs train-horizon causal floor: `24.72%` / `22.36%` / `14.35%` / `23.89%`.
- runtime easy degradation: `-25.63%`; switch rate: `58.81%`.
- exact replay: switch `True`, selected_xy max abs diff `0.0`.
- near@0.05 base/final/floor: `1.94%` / `1.38%` / `2.24%`.
- Paper wording: protected source-level group-consistency full-waypoint runtime policy is valid evidence, but ungated full-waypoint and global primary replacement remain blocked.

### Deployment Variant Boundary

- safety-sensitive default: `proximity_guard`.
- accuracy-priority diagnostic: `no_proximity_guard`.
- source-level full-waypoint runtime candidate: `group_consistency_full_waypoint_runtime`.
- baseline mixing caveat: `True`.

### Source / Time / Metric Boundary

- conversion-ready targets: `0`; converted now: `0`; evaluated now: `0`.
- global metric/seconds claim allowed: `False`.
- global t100 deployable claim allowed: `False`.
- Paper wording: dataset-local/raw-frame only unless future source/legal/time calibration closes the blocker.

### Non-Claims

- Do not claim true 3D.
- Do not claim foundation world model.
- Do not claim global metric or seconds-level prediction.
- Do not claim Stage5C execution.
- Do not claim SMC readiness.
<!-- STAGE42_DR_POST_DQ_PAPER_REFRESH:END -->

<!-- STAGE42_EB_POST_EA_PAPER_REFRESH:START -->
## Stage42-EB Post-EA Paper Evidence Refresh

- source: `fresh_paper_refresh_from_stage42_dy_dz_ea`
- role: synchronize paper-ready artifacts after explicit physical consistency and dual-domain bootstrap evidence.
- This is a paper-package update from fresh Stage42-DY/DZ/EA evidence, not new training and not a threshold search.

### What Changed After EA

- scalar loss-family promotion remains blocked: best `proximity_occupancy_loss` all/t50/hard `25.51%` / `22.14%` / `23.74%`.
- explicit group-consistency is source-level promoted: all/t50/t100 raw/hard `24.72%` / `22.36%` / `14.35%` / `23.89%`.
- group-consistency delta vs Stage42-AM all/hard: `0.14%` / `0.14%`.
- near@0.05 is repaired from `1.94%` to `1.38%` in the DY checkpoint.

### Dual-Domain Evidence

- positive safe domains: `2`.
- UCY all/t50/hard: `35.58%` / `22.72%` / `33.78%`.
- TrajNet all/t50/hard: `32.07%` / `28.18%` / `31.29%`.

### Bootstrap Evidence

- bootstrap_n: `2000`.
- global all/t50/hard CI: `[32.56%, 33.23%]` / `[26.53%, 27.44%]` / `[31.51%, 32.26%]`; easy degradation CI `[-32.96%, -31.28%]`.
- UCY all/t50/hard CI: `[34.70%, 36.49%]` / `[21.38%, 24.18%]` / `[32.84%, 34.76%]`.
- TrajNet all/t50/hard CI: `[31.72%, 32.41%]` / `[27.72%, 28.61%]` / `[30.90%, 31.66%]`.
- near@0.05 final-base delta CI: `[-0.86%, -0.67%]`.

### Updated Paper Claim Boundary

- Supported: protected source-level group-consistency full-waypoint dynamics with UCY+TrajNet bootstrap-backed raw-frame evidence.
- Supported: explicit physical/group-consistency as a source-level full-waypoint repair route.
- Not supported as main claims: scalar loss weighting, goal/scene context, and neighbor/interaction context under current protocols.
- Not supported: ungated full-waypoint deployment or global primary full-waypoint replacement.
- Still forbidden: true 3D, foundation model, global metric/seconds-level claims, Stage5C execution, and SMC readiness.
<!-- STAGE42_EB_POST_EA_PAPER_REFRESH:END -->

<!-- STAGE42_EG_POST_EE_EF_PAPER_REFRESH:START -->
## Stage42-EG Post-EE/EF Paper Claim Refresh

- source: `fresh_paper_refresh_from_stage42_eb_ec_ee_ef`
- role: integrate context materiality and source terms gap evidence into the paper claim/gap matrix.
- This is a paper-package refresh, not new training, conversion, download, or threshold tuning.

### Main Claim Boundary After EE/EF

- Supported main claim: protected source-level group-consistency full-waypoint dynamics with dual-domain bootstrap evidence.
- Context main claim remains blocked: selected `baseline_plus_knn_graph` deltas all/t50/hard `0.000368` / `-0.000074` / `0.000424`, below threshold `0.01`.
- Source conversion remains blocked: conversion_ready_now `0`, converted/evaluated now `0` / `0`.
- Source unlock potential after terms: t50/t100 `10060` / `5696`, top targets `['ucy_crowd_original', 'eth_biwi_original', 'aerialmpt_or_other_topdown']`.
- Still forbidden: true 3D, foundation model, global metric/seconds-level claims, Stage5C execution, and SMC readiness.
<!-- STAGE42_EG_POST_EE_EF_PAPER_REFRESH:END -->

<!-- STAGE42_EO_POST_EM_EN_PAPER_REFRESH:START -->
## Stage42-EO Post-EM/EN Paper Package Refresh

- source: `fresh_paper_refresh_from_stage42_eg_em_en`
- role: propagate official-source/manual-terms blockers and floor-removability decisions into the paper package.
- This is a paper-package refresh, not new training, conversion, download, or threshold tuning.

### Source / Legal Boundary

- official/toolkit source candidates: `4` / `5`.
- manual terms required targets: `5`.
- auto_download_allowed_now: `0`; conversion_ready_now: `0`; converted/evaluated now: `0` / `0`.
- after-terms potential t50/t100 windows: `10060` / `5696`.
- Official links are not license acceptance; user must confirm terms, allowed use, local path, and source identity before conversion.

### Safety Floor Boundary

- floor_free_neural_deployable: `False`.
- global_floor_removal_allowed: `False`.
- teacher_floor_rollout_context_removal_allowed: `False`.
- safe_partial_floor_relaxation_available: `True` on `['t50_slice_relaxation::TrajNet|50', 't50_slice_relaxation::UCY|50']`.
- proximity_guard_required_for_safety_claim: `True`.

### Updated Paper Claim Boundary

- Supported: protected source-level group-consistency full-waypoint raw-frame 2.5D evidence.
- Supported only as narrow slice evidence: validation-backed t50 floor relaxation on mapped slices.
- Required: Stage37/teacher floor rollout context, deployment fallback floor, and proximity guard for safety-sensitive reporting.
- Blocked: source conversion without user terms/path/source identity; global floor-free neural; teacher-floor rollout context removal.
- Still forbidden: true 3D, foundation model, global metric/seconds-level claims, Stage5C execution, and SMC readiness.
<!-- STAGE42_EO_POST_EM_EN_PAPER_REFRESH:END -->

<!-- STAGE42_ER_POST_EQ_CONTEXT_CLAIM_REFRESH:START -->
## Stage42-ER Post-EQ Context Claim Refresh

- source: `fresh_post_eq_context_claim_refresh`
- role: integrate the fresh Stage42-EQ sequence+graph router result into the paper/gap/action package.
- This is not new training, conversion, download, or threshold tuning.
- Stage42-EQ best router: `baseline_plus_history_goal_neighbor` with all/t50/t100raw/hard deltas `0.01%` / `-0.02%` / `0.01%` / `0.02%`.
- context decision: `close_current_shallow_sequence_graph_context_protocol`.
- independent context main claim allowed: `False`.
- closed protocols: `['context_gain_router', 'sequence_residual_context', 'graph_residual_context', 'sequence_graph_context_router']`.
- Paper wording: sequence/graph/goal/neighbor context remains auxiliary or diagnostic under current protocols, not an independent main contribution.
- Next primary route: source/legal/time conversion and stronger joint occupancy / interaction-constraint targets, not repeating shallow context routers.
- Boundary: raw-frame/dataset-local 2.5D only; no true 3D, no foundation, no metric/seconds, no Stage5C, no SMC.
<!-- STAGE42_ER_POST_EQ_CONTEXT_CLAIM_REFRESH:END -->

<!-- STAGE42_ES_INTERACTION_OCCUPANCY_TARGET_SELECTION:START -->
## Stage42-ES Interaction / Occupancy Target Selection

- source: `fresh_stage42_interaction_occupancy_target_selection`
- role: fresh-reruns scalar proximity/occupancy loss and explicit group-consistency repair to choose the next interaction/occupancy target.
- selected target family: `explicit_group_consistency_repair`; decision `continue_with_explicit_group_consistency_interaction_target`.
- scalar proximity/occupancy target all/t50/hard/easy: `25.51%` / `22.14%` / `23.74%` / `-29.23%`.
- explicit group-consistency target all/t50/t100raw/hard/easy: `24.72%` / `22.36%` / `14.35%` / `23.89%` / `-25.63%`.
- group near@0.05 base/final: `1.94%` / `1.38%`.
- Claim boundary: protected source-level raw-frame full-waypoint evidence only; not true 3D, not foundation, not metric/seconds, no Stage5C, no SMC.
<!-- STAGE42_ES_INTERACTION_OCCUPANCY_TARGET_SELECTION:END -->

<!-- STAGE42_ET_GROUP_CONSISTENCY_TARGET_ABLATION:START -->
## Stage42-ET Group-Consistency Target Ablation

- source: `fresh_stage42_group_consistency_target_ablation`
- role: tests whether the Stage42-ES selected group-consistency target depends on the real source/frame/horizon multi-agent grouping.
- source/frame/horizon all/t50/t100raw/hard/easy: `24.72%` / `22.36%` / `14.35%` / `23.89%` / `-25.63%`.
- agent-isolated control all/t50/hard/easy: `24.58%` / `22.02%` / `23.75%` / `-25.66%`.
- source/frame/horizon vs isolated hard increment `0.14%`; own-base near@0.05 reduction `0.55%`.
- decision: `keep_source_frame_horizon_group_consistency_target`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_ET_GROUP_CONSISTENCY_TARGET_ABLATION:END -->

<!-- STAGE42_EU_GROUP_CONSISTENCY_CONSTRAINT_TRAINING:START -->
## Stage42-EU Group-Consistency Constraint Training

- source: `fresh_stage42_group_consistency_constraint_training`
- role: trains full-waypoint dynamics with source/frame/horizon group-risk weighted losses, then applies validation-selected group repair.
- selected training variant: `group_unsafe_weighted` with `stage42_am_features` lambda `10.0`.
- test all/t50/t100raw/hard/easy: `22.81%` / `22.35%` / `12.68%` / `21.97%` / `-23.91%`.
- delta vs Stage42-DI all/hard/easy: `-1.90%` / `-1.91%` / `1.72%`.
- near@0.05 base/final: `1.88%` / `1.33%`.
- decision: `group_constraint_training_not_enough_keep_stage42_di_or_cq_floor`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_EU_GROUP_CONSISTENCY_CONSTRAINT_TRAINING:END -->

<!-- STAGE42_EV_CONSTRAINT_AWARE_COMPOSER:START -->
## Stage42-EV Constraint-Aware Composer

- source: `fresh_stage42_constraint_aware_composer`
- role: validation-only composer over floor / Stage42-AM / Stage42-DI / Stage42-EU by domain, horizon, and group-risk buckets.
- selected composer mode: `domain_horizon`.
- test all/t50/t100raw/hard/easy: `24.71%` / `22.35%` / `14.35%` / `23.88%` / `-25.10%`.
- delta vs Stage42-DI all/hard/easy: `-0.00%` / `-0.00%` / `0.53%`.
- near@0.05 base/final: `1.94%` / `1.37%`.
- decision: `constraint_aware_composer_positive_but_keep_stage42_di`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_EV_CONSTRAINT_AWARE_COMPOSER:END -->

<!-- STAGE42_EW_ADAPTIVE_GROUP_REPAIR:START -->
## Stage42-EW Adaptive Group Repair

- source: `fresh_stage42_adaptive_group_repair`
- role: validation-only adaptive selection over Stage42-DI repair candidates by global / domain+horizon / domain+horizon+risk slices.
- selected mode: `domain_horizon`.
- test all/t50/t100raw/hard/easy: `24.70%` / `22.36%` / `14.35%` / `23.88%` / `-25.64%`.
- delta vs Stage42-DI all/hard/easy: `-0.01%` / `-0.01%` / `-0.01%`.
- near@0.05 base/final: `1.94%` / `1.44%`.
- mixed group selection rate: `0.00%`.
- decision: `stage42_ew_adaptive_group_repair_positive_not_promoted`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_EW_ADAPTIVE_GROUP_REPAIR:END -->

<!-- STAGE42_EX_GROUP_LEVEL_RISK_REPAIR:START -->
## Stage42-EX Group-Level Risk Repair

- source: `fresh_stage42_group_level_risk_repair`
- role: validation-only adaptive repair where risk is aggregated to source/frame/horizon groups before selecting repair candidates.
- selected mode: `domain_horizon`.
- test all/t50/t100raw/hard/easy: `24.70%` / `22.36%` / `14.35%` / `23.88%` / `-25.64%`.
- delta vs Stage42-DI all/hard/easy: `-0.01%` / `-0.01%` / `-0.01%`.
- near@0.05 base/final: `1.94%` / `1.44%`.
- mixed group selection rate: `0.00%`.
- decision: `stage42_ex_group_level_risk_repair_positive_not_promoted`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_EX_GROUP_LEVEL_RISK_REPAIR:END -->

<!-- STAGE42_EY_CONTINUOUS_GROUP_RISK_REPAIR:START -->
## Stage42-EY Continuous Group-Risk Repair

- source: `fresh_stage42_continuous_group_risk_repair`
- role: validation-only continuous group-risk bucket repair over Stage42-DI repair candidates.
- selected mode: `domain_horizon`.
- test all/t50/t100raw/hard/easy: `24.70%` / `22.36%` / `14.35%` / `23.88%` / `-25.64%`.
- delta vs Stage42-DI all/hard/easy: `-0.01%` / `-0.01%` / `-0.01%`.
- near@0.05 base/final: `1.94%` / `1.44%`.
- mixed group selection rate: `0.00%`.
- decision: `stage42_ey_continuous_group_risk_repair_positive_not_promoted`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_EY_CONTINUOUS_GROUP_RISK_REPAIR:END -->

<!-- STAGE42_EZ_TEMPORAL_GROUP_REPEL_REPAIR:START -->
## Stage42-EZ Temporal Group-Repel Repair

- source: `fresh_stage42_temporal_group_repel_repair`
- role: tests temporal weighting for group-repel offsets after Stage42-EW/EX/EY risk-bucket repairs failed to beat Stage42-DI.
- selected candidate: `{'mode': 'temporal_repel', 'temporal_kind': 'tail', 'gamma': 1.0, 'direction_mode': 'nearest_current', 'min_sep': 0.12, 'margin': 0.0, 'strength': 0.25}`.
- gate: `17 / 18`; verdict `stage42_ez_temporal_group_repel_repair_positive_not_promoted`.
- test all/t50/t100raw/hard/easy: `24.73%` / `22.40%` / `14.35%` / `23.89%` / `-25.64%`.
- delta vs Stage42-DI all/t50/t100raw/hard/easy: `0.01%` / `0.04%` / `0.00%` / `0.00%` / `-0.01%`.
- near@0.05 base/final: `1.94%` / `1.51%`.
- decision: `temporal_group_repel_not_enough_keep_stage42_di_or_cq_floor`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_EZ_TEMPORAL_GROUP_REPEL_REPAIR:END -->

<!-- STAGE42_FA_WAYPOINTWISE_GROUP_REPEL_REPAIR:START -->
## Stage42-FA Waypoint-Wise Group-Repel Repair

- source: `fresh_stage42_waypointwise_group_repel_repair`
- role: tests per-waypoint group-consistency offsets after Stage42-EZ temporal single-direction repair failed proximity promotion.
- selected candidate: `{'mode': 'waypointwise_repel', 'min_sep': 0.12, 'strength': 0.2, 'temporal_kind': 'sqrt_tail', 'gamma': 1.0, 'smooth': True, 'cap_scale': 0.75}`.
- gate: `15 / 17`; verdict `stage42_fa_waypointwise_group_repel_repair_positive_not_promoted`.
- test all/t50/t100raw/hard/easy: `24.61%` / `22.05%` / `14.36%` / `23.77%` / `-25.67%`.
- delta vs Stage42-DI all/t50/t100raw/hard/easy: `-0.11%` / `-0.31%` / `0.02%` / `-0.11%` / `-0.03%`.
- near@0.05 base/final: `1.94%` / `1.21%`.
- decision: `waypointwise_group_repel_not_enough_keep_stage42_di_or_cq_floor`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FA_WAYPOINTWISE_GROUP_REPEL_REPAIR:END -->

<!-- STAGE42_FB_PROXIMITY_PARETO_COMPOSER:START -->
## Stage42-FB Proximity Pareto Composer

- source: `fresh_stage42_proximity_pareto_composer`
- role: validation-only composer between Stage42-DI accuracy policy and Stage42-FA proximity-safety policy.
- selected candidate: `{'mode': 'group_di_near_fa_safer', 'threshold': 0.05, 'margin': 0.0}`.
- gate: `14 / 16`; verdict `stage42_fb_proximity_pareto_composer_positive_not_promoted`.
- test all/t50/t100raw/hard/easy: `24.65%` / `22.19%` / `14.35%` / `23.82%` / `-25.64%`.
- delta vs Stage42-DI all/t50/t100raw/hard/easy: `-0.07%` / `-0.18%` / `0.00%` / `-0.07%` / `-0.01%`.
- near@0.05 final/use_fa_rate: `1.10%` / `9.34%`.
- decision: `proximity_pareto_composer_not_enough_keep_stage42_di_or_cq_floor`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FB_PROXIMITY_PARETO_COMPOSER:END -->

<!-- STAGE42_FC_OBJECTIVE_LEVEL_PROXIMITY_TRAINING:START -->
## Stage42-FC Objective-Level Proximity Training

- source: `fresh_stage42_objective_level_proximity_training`
- role: moves proximity/group-interaction signal from post-hoc repair into supervised full-waypoint training objective.
- selected objective: `label_proximity_objective`; feature mode `stage42_am_features`; lambda `10.0`.
- gate: `22 / 23`; verdict `stage42_fc_objective_level_proximity_training_positive_not_promoted`.
- test all/t50/t100raw/hard/easy: `26.37%` / `23.01%` / `14.02%` / `24.76%` / `-31.10%`.
- delta vs Stage42-DI all/hard/near005: `1.66%` / `0.87%` / `0.48%`.
- decision: `objective_level_training_not_enough_keep_stage42_di_or_cq_floor`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FC_OBJECTIVE_LEVEL_PROXIMITY_TRAINING:END -->

<!-- STAGE42_FD_SAFETY_AWARE_JOINT_OBJECTIVE:START -->
## Stage42-FD Safety-Aware Joint Objective Training

- source: `fresh_stage42_safety_aware_joint_objective_training`
- role: tests whether FA safety-teacher regularization inside the training objective can break the FC accuracy/proximity tradeoff.
- selected objective: `fc_label_proximity_control`; feature mode `stage42_am_features`; lambda `100.0`; teacher alpha `0.0`.
- gate: `22 / 26`; verdict `stage42_fd_safety_aware_joint_objective_positive_not_promoted`.
- test all/t50/t100raw/hard/easy: `26.33%` / `22.70%` / `14.02%` / `24.69%` / `-31.11%`.
- delta vs Stage42-FC all/hard/near005: `-0.04%` / `-0.07%` / `0.01%`.
- delta vs Stage42-DI all/hard/near005: `1.62%` / `0.80%` / `0.48%`.
- decision: `safety_aware_objective_not_enough_keep_stage42_di_or_cq_floor`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FD_SAFETY_AWARE_JOINT_OBJECTIVE:END -->

<!-- STAGE42_FE_CONSTRAINED_FC_SAFETY_COMPOSER:START -->
## Stage42-FE Constrained FC/Safety Composer

- source: `fresh_stage42_constrained_fc_safety_composer`
- role: validation-only constrained composer from high-accuracy Stage42-FC to DI/FA/FB safety fallbacks.
- selected candidate: `{'mode': 'fc_to_safety', 'fallback': 'di', 'scope': 'row', 'threshold': 0.05, 'margin': 0.0025}`.
- gate: `19 / 19`; verdict `stage42_fe_constrained_fc_safety_composer_pass_promotable`.
- test all/t50/t100raw/hard/easy: `26.41%` / `23.15%` / `14.01%` / `24.81%` / `-31.06%`.
- delta vs FC all/hard/near005: `0.04%` / `0.05%` / `-0.54%`.
- delta vs DI all/hard/near005: `1.69%` / `0.92%` / `-0.06%`.
- decision: `promote_stage42_fe_constrained_fc_safety_composer`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FE_CONSTRAINED_FC_SAFETY_COMPOSER:END -->

<!-- STAGE42_FF_FE_POLICY_FREEZE_REPLAY:START -->
## Stage42-FF FE Policy Freeze / Bootstrap / Replay

- source: `fresh_stage42_fe_policy_freeze_replay`
- role: freeze Stage42-FE constrained FC/safety composer and add 2000-bootstrap plus exact replay evidence.
- gate: `23 / 23`; verdict `stage42_ff_fe_policy_freeze_replay_pass`.
- frozen policy hash: `a78db26aa155b38799f5b866f32a2d205018adf2054d9409a016da3163328dff`.
- replay all/t50/t100raw/hard/easy: `26.41%` / `23.15%` / `14.01%` / `24.81%` / `-31.06%`.
- bootstrap lows all/t50/t100raw/hard: `26.08%` / `22.71%` / `13.46%` / `24.46%`.
- exact replay max metric/diagnostic diff: `0.0` / `0.0`.
- Boundary: frozen protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FF_FE_POLICY_FREEZE_REPLAY:END -->

<!-- STAGE42_FG_FE_SOURCE_ROBUSTNESS:START -->
## Stage42-FG FE Source / Domain / Horizon Robustness Audit

- source: `fresh_stage42_fe_source_robustness_audit`
- role: audit frozen Stage42-FE/FF across domain/source/horizon/scene slices without retraining or threshold reselection.
- gate: `11 / 12`; verdict `stage42_fg_fe_source_robustness_partial`.
- robust domains: `['TrajNet']`.
- weak domain-horizon slices: `['TrajNet|100', 'UCY|10', 'UCY|25', 'UCY|50', 'UCY|100']`.
- weak sources: `['TrajNet/Train/crowds/crowds_zara03.txt']`.
- broad uniform source claim allowed: `False`.
- Boundary: protected source-level raw-frame 2.5D audit; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FG_FE_SOURCE_ROBUSTNESS:END -->

<!-- STAGE42_FH_UCY_SUPPORTED_FE_COMPOSER:START -->
## Stage42-FH UCY-Supported FE Composer

- source: `fresh_stage42_ucy_supported_fe_composer`
- role: repair Stage42-FG UCY fallback-only weakness by adding train-only UCY internal validation before FE composer selection.
- gate: `20 / 20`; verdict `stage42_fh_ucy_supported_fe_composer_pass`.
- positive safe domains: `['TrajNet', 'UCY']`; weak domains: `[]`.
- all/t50/t100raw/hard/easy: `34.98%` / `28.97%` / `20.57%` / `33.10%` / `-36.91%`.
- decision: `promote_stage42_fh_ucy_supported_fe_composer`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FH_UCY_SUPPORTED_FE_COMPOSER:END -->

<!-- STAGE42_FI_FH_POLICY_FREEZE_REPLAY:START -->
## Stage42-FI FH Policy Freeze / Bootstrap / Replay

- source: `fresh_stage42_fh_policy_freeze_replay`
- role: freeze Stage42-FH UCY-supported FE composer and add 2000-bootstrap plus exact replay evidence.
- gate: `25 / 25`; verdict `stage42_fi_fh_policy_freeze_replay_pass`.
- frozen policy hash: `f1f6e0636167fae8721a3f7195f188dcbe1a83194b04fa0625b378ad38b5aed6`.
- replay all/t50/t100raw/hard/easy: `34.98%` / `28.97%` / `20.57%` / `33.10%` / `-36.91%`.
- bootstrap lows all/t50/t100raw/hard: `34.62%` / `28.46%` / `19.96%` / `32.73%`.
- exact replay max metric/diagnostic diff: `0.0` / `0.0`.
- dual-domain support: UCY `True`, TrajNet `True`.
- Boundary: frozen protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FI_FH_POLICY_FREEZE_REPLAY:END -->
