# Stage42 Model Card

## Model

M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under Stage37/teacher floor.

## Intended Use

Research evaluation for dataset-local raw-frame top-down multi-agent trajectory/world-state prediction and failure/hard-case diagnostics.

## Not Intended For

- metric or seconds-level physical deployment
- true 3D world modeling
- large-scale foundation model claims
- autonomous deployment without dataset/domain validation
- Stage5C latent generative execution or SMC

## Performance Summary

- protected external all: `0.2103`
- protected external t50: `0.1365`
- protected external t100 raw-frame diagnostic: `0.1469`
- protected external hard/failure: `0.2038`
- protected easy degradation: `0.0000`
- full-waypoint ADE all/t50: `0.1858` / `0.1480`

## Safety

The Stage37/teacher floor is required for current deployment. Ungated neural is explicitly rejected.

<!-- STAGE42_AC_REFRESH:START -->
## Stage42-AC Latest Evidence Refresh

- source: `fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-AB is now included as auxiliary-head evidence.
- Auxiliary-head conclusion: mixed / partial; not a uniform main contribution claim.
- Stage5C and SMC remain disabled.

### Model-Card Update

Interaction/occupancy/physical heads are present in the full-waypoint model interface, but Stage42-AB shows they should be described as auxiliary diagnostics/regularizers with mixed evidence, not as uniformly beneficial deployable heads.
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

<!-- STAGE42_CM_FULL_WAYPOINT_BRIDGE_SHAPE_AUDIT:START -->
## Stage42-CM Endpoint Bridge / Full-Waypoint Shape Audit

- source: `fresh_synthesis_from_stage42_full_waypoint_artifacts`
- scope: protected dataset-local raw-frame 2.5D full-waypoint evidence boundary.
- Endpoint-only FDE is diagnostic; endpoint success cannot be counted as full-waypoint world-state success.
- Protected full-waypoint sequence dynamics has horizon/full-waypoint evidence, but the endpoint linear bridge remains stronger on all-ADE.
- Graph/group consistency has positive protected metrics but still carries a proximity caveat and is not an independent neighbor/interaction main claim after Stage42-CK.
- Stage5C remains unexecuted and SMC remains disabled.

### Key Deltas

- protected full-waypoint minus composite-tail linear bridge: all `-2.45%`, t50 `1.15%`, t100 raw diagnostic `8.16%`, hard `-0.87%`.
- graph/group minus protected full-waypoint: all `3.66%`, t50 `0.29%`, hard `2.89%`, collision_delta_005 `0.00829083972266037`.

### Evidence Rows

| variant | status | all | t50 | t100 diag | hard | easy | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `endpoint_only_final_fde` | `diagnostic_only_not_full_waypoint` | 0.00% | 0.00% | 0.00% | 0.00% | -0.00% | Endpoint-only final FDE is not a full-waypoint world-state model. |
| `m3w_neural_v1_composite_tail_linear_bridge` | `deployable_endpoint_linear_bridge_floor` | 21.03% | 13.65% | 14.69% | 20.38% | -14.51% | Current protected endpoint dynamics projected through a linear bridge. |
| `full_waypoint_transformer_protected` | `protected_full_waypoint_positive_two_domains` | 18.58% | 14.80% | 22.86% | 19.52% | -0.00% | Actual full-waypoint sequence model under protected switch policy. |
| `ungated_full_waypoint_transformer` | `diagnostic_unsafe_not_deployable` | 29.66% | 21.52% | 35.92% | 32.94% | 124.59% | Ungated neural/full-waypoint output has unsafe easy degradation. |
| `graph_interaction_group_consistency` | `protected_positive_with_proximity_caveat` | 22.24% | 15.09% | 23.02% | 22.41% | 0.00% | Protected graph/group policy positive, but collision_delta_vs_floor_005=0.00829083972266037; CK blocks graph as independent main claim. |
| `unified_row_level_full_waypoint_cache` | `row_level_full_waypoint_three_domain_positive` | 9.00% | 6.11% | 8.15% | 9.37% | 0.11% | Unified row-level cache merges verified external full-waypoint policy sources across ETH_UCY, TrajNet, and UCY. |
| `ucy_endpoint_to_full_linear_bridge` | `failed_blocker` | n/a | n/a | n/a | n/a | n/a | Stage41 pure-UCY endpoint residual is positive on endpoint FDE, but linear endpoint-to-waypoint interpolation is negative on Stage42 full-waypoint validation and UCY test. Endpoint success cannot be counted as full-waypoint world-state success. |

### Claim Boundary

- Supported: protected full-waypoint raw-frame evidence exists, especially for horizon/full-waypoint slices and unified row-level three-domain package.
- Supported with caveat: graph/group consistency can be useful inside protected policies, but current source-level kNN graph expert did not become a main contribution.
- Rejected: endpoint-only success as full-waypoint success; ungated full-waypoint neural deployment; true 3D; foundation; metric/seconds-level; Stage5C; SMC.
<!-- STAGE42_CM_FULL_WAYPOINT_BRIDGE_SHAPE_AUDIT:END -->

<!-- STAGE42_CN_BRIDGE_SHAPE_COMPOSER:START -->
## Stage42-CN Bridge / Shape Composer Audit

- source: `fresh_synthesis_from_stage42_cm_j_x_artifacts`
- scope: validation-only composer feasibility for endpoint-linear bridge vs full-waypoint shape heads.
- conclusion: keep endpoint-linear bridge as deployable all-ADE floor; use full-waypoint heads only as auxiliary horizon evidence until a common validation-aligned row-level composer exists.
- blocker: common validation endpoint-vs-full-waypoint row cache is missing, so no new bridge/shape deployment switch is allowed.
- Stage5C remains unexecuted and SMC remains disabled.

### Candidate Summary

| candidate | status | all | t50 | t100 diag | hard | easy | role |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `endpoint_linear_bridge_floor` | `current_deployable_all_ade_floor` | 21.03% | 13.65% | 14.69% | 20.38% | -14.51% | Current M3W-Neural v1 protected endpoint dynamics projected through endpoint-linear waypoint bridge. |
| `protected_full_waypoint_sequence` | `protected_full_waypoint_horizon_auxiliary` | 18.58% | 14.80% | 22.86% | 19.52% | -0.00% | Actual full-waypoint sequence model; useful on t50/t100 raw-frame but not an all-ADE replacement. |
| `stage42j_static_gated` | `validation_selected_full_waypoint_shape_candidate` | 3.62% | 3.69% | 2.67% | 3.97% | 0.00% | Stage42-J uses validation-only domain/horizon static gating over cached full-waypoint checkpoints. |
| `stage42j_static_alpha025` | `validation_selected_full_waypoint_shape_candidate` | 3.52% | 3.44% | 3.01% | 3.87% | 0.00% | Stage42-J uses validation-only domain/horizon static gating over cached full-waypoint checkpoints. |
| `stage42j_no_static` | `validation_selected_full_waypoint_shape_candidate` | 1.15% | 1.99% | 1.41% | 1.29% | 0.00% | Stage42-J uses validation-only domain/horizon static gating over cached full-waypoint checkpoints. |
| `stage42x_unified_row_level_full_waypoint_cache` | `row_level_full_waypoint_three_domain_positive_auxiliary` | 9.00% | 6.11% | 8.15% | 9.37% | 0.11% | Unified row-level full-waypoint cache is positive but below the current endpoint-linear bridge floor on all/t50/hard. |
| `ungated_full_waypoint_sequence` | `diagnostic_unsafe_not_deployable` | 29.66% | 21.52% | 35.92% | 32.94% | 124.59% | Ungated full-waypoint neural output is unsafe because easy degradation is far above the deployment limit. |

### Deployment Boundary

- selected deployment policy: `keep_endpoint_linear_bridge_floor_with_full_waypoint_auxiliary_reporting`
- deployable bridge/shape composer available now: `False`
- next required evidence: Build common validation-aligned endpoint-linear-vs-full-waypoint row cache before any deployment switch.
<!-- STAGE42_CN_BRIDGE_SHAPE_COMPOSER:END -->

<!-- STAGE42_CO_COMMON_VALIDATION_BRIDGE_SHAPE_COMPOSER:START -->
## Stage42-CO Common Validation Bridge / Shape Composer

- source: `fresh_common_validation_eval_from_cached_verified_checkpoints`
- common validation/test row alignment is verified for endpoint-linear bridge and full-waypoint sequence.
- policy is selected on validation rows only; test is evaluated once.
- composer vs endpoint-linear bridge ADE: all `3.02%`, t50 `1.50%`, t100 raw diagnostic `6.12%`, hard `3.28%`, easy `0.25%`.
- composer vs strongest causal floor ADE: all `23.41%`, t50 `14.95%`, t100 raw diagnostic `19.91%`, hard `23.00%`.
- use_full_rate: `21.35%`.
- alignment: `True`.
- claim boundary: no true 3D, no foundation, no metric/seconds-level, no Stage5C, no SMC.
<!-- STAGE42_CO_COMMON_VALIDATION_BRIDGE_SHAPE_COMPOSER:END -->

<!-- STAGE42_CP_COMMON_VALIDATION_COMPOSER_SAFETY:START -->
## Stage42-CP Common Validation Composer Safety / Bootstrap

- source: `fresh_joint_safety_bootstrap_from_stage42_co_policy`
- scope: Stage42-CO validation-selected composer, test evaluated once.
- bootstrap vs endpoint-linear all CI: `[2.64%, 3.37%]`.
- bootstrap vs endpoint-linear t50 CI: `[0.90%, 2.09%]`.
- bootstrap vs endpoint-linear t100 raw diagnostic CI: `[5.39%, 6.94%]`.
- bootstrap vs endpoint-linear hard/failure CI: `[2.90%, 3.68%]`.
- near-collision@0.05 delta vs endpoint-linear: `0.34%`.
- near-collision@0.05 delta vs strongest floor: `-0.05%`.
- claim boundary: still dataset-local/raw-frame 2.5D; no metric/seconds-level, no Stage5C, no SMC.
<!-- STAGE42_CP_COMMON_VALIDATION_COMPOSER_SAFETY:END -->
