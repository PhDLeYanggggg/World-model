# Stage42 Method Draft

## Problem

Given past-only multi-agent history, neighbor context, goal/prototype context, and a strongest causal/Stage37 teacher floor, predict future endpoint and full-waypoint world-state trajectories under strict no-leakage constraints.

## Inputs

- past-only history windows and causal velocities
- neighbor/interaction/group-consistency features
- domain/horizon metadata
- train-only goal/prototype features where available
- Stage37/teacher floor rollout and proposal scores

No future endpoint, future waypoint, central velocity, or test endpoint goal is used as inference input.

## Model

The deployable path is a composite-tail safe-switch bounded neural dynamics policy under the Stage37/teacher floor. It combines a validation-selected teacher repaired switch with a small bounded tail alpha for confident neural proposals. Stage42-C additionally evaluates a protected full-waypoint sequence model on reconstructed future waypoint labels.

## Safety

Stage42-E evaluates internal self-gates, uncertainty gates, harm gates, conformal-style risk gates, teacher-prob gates, and bounded residual blends. The current deployable conclusion is that the Stage37/teacher floor remains necessary. Ungated neural improves raw error but fails safety.

## Claim Boundary

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space；external 是 dataset-local / unverified weak metric diagnostic。
- t+50 / t+100 是 raw-frame horizon，不能写成 seconds-level。
- global metric/time claims 仍不允许；TGSIM 只能作为 traffic diagnostic，不是 pedestrian official claim。
- self-audited / visual-prior labels 不是 human gold。
- Stage5C latent generative 未执行。
- SMC 未启用。

<!-- STAGE42_AC_REFRESH:START -->
## Stage42-AC Latest Evidence Refresh

- source: `fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-AB is now included as auxiliary-head evidence.
- Auxiliary-head conclusion: mixed / partial; not a uniform main contribution claim.
- Stage5C and SMC remain disabled.

### Method Update

The deployable full-waypoint model should treat interaction/occupancy/physical heads as auxiliary diagnostics. They may regularize t50/FDE@50, but the current evidence does not justify making them a central uniform contribution.
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

<!-- STAGE42_CW_RUNTIME_REPLAY_REFRESH:START -->
## Stage42-CW Runtime Replay Paper / Reproducibility Refresh

- source: `fresh_synthesis_from_stage42_cv_runtime_batch_replay`
- role: paper-ready deployment reproducibility evidence.
- Stage42-CV gate: `25 / 25`; verdict `stage42_cv_batch_runtime_replay_pass`.
- frozen policy hash: `4af6536f86499d5b39efa535bb81978398586d65746bb983571b642af7c92d59`.
- validation/test replay rows: `53256` / `55528`.
- exact runtime replay: validation `True`, test `True`.
- selected_xy / ADE / FDE max diff vs original CQ guard on test: `0.0` / `0.0` / `0.0`.
- test ADE vs endpoint-linear all/t50/t100 raw/hard: `1.77%` / `1.07%` / `3.48%` / `1.93%`.
- easy degradation: `0.25%`; switch rate: `16.96%`.
- near-collision@0.05 delta vs endpoint-linear: `-0.06%`; jagged-rate delta: `0.00%`.
- The guard's second proximity input is the validation-selected base composer candidate rollout group min-distance, not future labels.
- This refresh does not create a new metric/seconds/3D/foundation claim; it only strengthens deployable policy reproducibility under protected dataset-local/raw-frame 2.5D boundaries.
- Stage5C remains unexecuted and SMC remains disabled.
<!-- STAGE42_CW_RUNTIME_REPLAY_REFRESH:END -->

<!-- STAGE42_DJ_FROZEN_GROUP_CONSISTENCY_POLICY:START -->
## Stage42-DJ Frozen Group-Consistency Full-Waypoint Policy

- source: `fresh_policy_freeze_from_stage42_di`
- role: freeze the Stage42-DI promoted group-consistency full-waypoint repair as a reproducible policy artifact.
- repair uses predicted rollout geometry and source/frame/horizon group keys only; future waypoints remain labels/eval only.
- policy artifact: `outputs/stage42_long_research/frozen_group_consistency_full_waypoint_policy_stage42_policy.json`
- policy hash: `617ef9952b1439f3678318129a4979c7a171f2ba882742cd18acd46c5ae92141`
- test vs train-horizon causal floor ADE: all `24.72%`, t50 `22.36%`, t100 raw `14.35%`, hard `23.89%`, easy `-25.63%`.
- delta vs Stage42-AM all/t50/hard: `0.14%` / `0.35%` / `0.14%`.
- near@0.05 base/final/floor: `1.94%` / `1.38%` / `2.24%`.
- claim boundary: still protected dataset-local/raw-frame 2.5D; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_DJ_FROZEN_GROUP_CONSISTENCY_POLICY:END -->

<!-- STAGE42_DK_GROUP_CONSISTENCY_POLICY_REPLAY:START -->
## Stage42-DK Group-Consistency Policy Replay

- source: `fresh_replay_from_frozen_group_consistency_policy_artifact`
- role: replay the Stage42-DJ frozen group-consistency full-waypoint policy artifact against Stage42-DI/DJ source evidence.
- replay performs no retraining, no threshold reselection, and no test tuning.
- policy artifact: `outputs/stage42_long_research/frozen_group_consistency_full_waypoint_policy_stage42_policy.json`
- policy hash: `617ef9952b1439f3678318129a4979c7a171f2ba882742cd18acd46c5ae92141`
- gate: `34 / 34`; verdict `stage42_dk_group_consistency_policy_replay_pass`.
- replayed ADE vs train-horizon causal floor: all `24.72%`, t50 `22.36%`, t100 raw `14.35%`, hard `23.89%`, easy `-25.63%`.
- replayed near@0.05 base/final/floor: `1.94%` / `1.38%` / `2.24%`.
- claim boundary: still protected dataset-local/raw-frame 2.5D; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_DK_GROUP_CONSISTENCY_POLICY_REPLAY:END -->

<!-- STAGE42_DL_GROUP_CONSISTENCY_RUNTIME_POLICY:START -->
## Stage42-DL Group-Consistency Runtime Policy API

- source: `fresh_runtime_api_from_frozen_group_consistency_policy_artifact`
- role: expose Stage42-DJ/DK frozen group-consistency full-waypoint repair as a callable runtime policy.
- real batch replay uses reconstructed Stage42-DI source-level test rows and checks exact selected trajectory replay.
- policy artifact: `outputs/stage42_long_research/frozen_group_consistency_full_waypoint_policy_stage42_policy.json`
- policy hash: `617ef9952b1439f3678318129a4979c7a171f2ba882742cd18acd46c5ae92141`
- gate: `30 / 30`; verdict `stage42_dl_group_consistency_runtime_policy_pass`.
- replayed ADE vs train-horizon causal floor: all `24.72%`, t50 `22.36%`, t100 raw `14.35%`, hard `23.89%`, easy `-25.63%`.
- replayed near@0.05 base/final/floor: `1.94%` / `1.38%` / `2.24%`.
- claim boundary: still protected dataset-local/raw-frame 2.5D; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_DL_GROUP_CONSISTENCY_RUNTIME_POLICY:END -->

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
