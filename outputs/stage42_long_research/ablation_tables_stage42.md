# Stage42 Ablation Tables

| ablation | source | status | all | t50 | hard/failure | easy | interpretation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `no_neural_tail_use_teacher_floor_only` | `fresh_run` | `positive_safe` | 0.2036 | 0.1312 | 0.1966 | -0.1445 | Removing the composite-tail neural safe switch leaves the Stage37/teacher floor; positive protected-minus-teacher deltas indicate neural tail contribution. |
| `no_safe_floor_use_ungated_endpoint_neural` | `fresh_run` | `negative_unsafe` | 0.2966 | 0.2152 | 0.3294 | 1.2459 | Ungated endpoint neural is a no-fallback safety ablation; it can improve raw all but is not deployable if easy degradation exceeds 2%. |
| `oracle_floor_vs_neural_diagnostic` | `fresh_run` | `positive_safe` | 0.4222 | 0.3452 | 0.4211 | -0.2999 | Diagnostic oracle uses future labels only to measure remaining headroom; it is not a deployable model. |
| `no_full_waypoint_sequence_use_endpoint_linear_bridge` | `fresh_run` | `positive_safe` | 0.2103 | 0.1365 | 0.2038 | -0.1451 | Endpoint-linear bridge removes the full-waypoint sequence model. delta_vs_reference is ablation-minus-protected: negative t50/t100 deltas mean the full-waypoint model helps those horizons, while positive all-delta means endpoint-linear remains stronger on all-ADE. |
| `no_safe_floor_use_ungated_full_waypoint` | `fresh_run` | `negative_unsafe` | 0.2966 | 0.2152 | 0.3294 | 1.2459 | Ungated full-waypoint neural is a no-fallback safety ablation; it remains diagnostic if easy degradation is unsafe. |
| `no_composite_tail_use_teacher_linear_bridge` | `fresh_run` | `positive_safe` | 0.2036 | 0.1312 | 0.1966 | -0.1445 | Teacher linear bridge is the pre-composite floor in waypoint space; protected full-waypoint must improve without easy harm. |
| `no_history` | `cached_verified` | `complete` | n/a | n/a | n/a | n/a | Masks history/static causal feature group after policy freeze; proves coverage for no-history/static ablation, not a claim that every history-derived scalar is useless. |
| `no_neighbor` | `cached_verified` | `complete` | n/a | n/a | n/a | n/a | Neighbor/interaction masking is audited; group/neighbor features are especially important for the safety head and t100/hard slices. |
| `no_scene_goal` | `cached_verified` | `complete` | n/a | n/a | n/a | n/a | Scene/goal proxy coverage exists. Current deployable trajectory path keeps route/physical mostly diagnostic; route/physical heads are not main trajectory deployment claims. |
| `no_interaction` | `cached_verified` | `complete` | n/a | n/a | n/a | n/a | Interaction/group-consistency features have explicit ablations and are necessary for guarded deployment; without them raw neural remains less safe. |
| `no_jepa` | `cached_verified` | `complete_with_same_protocol_negative_evidence` | n/a | n/a | n/a | n/a | JEPA is explicitly disabled from the deployable path because audited JEPA variants were non-collapse but did not give deployable downstream lift. Stage41 now also records same-protocol pure-Transformer/no-JEPA attempts as negative or fallback-only, so the current positive path is protected endpoint neural dynamics rather than JEPA/Transformer purity. |
| `no_transformer` | `cached_verified` | `complete_with_same_protocol_negative_evidence` | n/a | n/a | n/a | n/a | Stage41 same-protocol JEPA-only/no-Transformer attempts are negative or unsafe, so no-Transformer is covered as negative architecture evidence. This is not a claim that JEPA contributes to the deployable path; it is why JEPA remains diagnostic-only. |
| `no_fallback` | `cached_verified` | `complete` | n/a | n/a | n/a | n/a | No-fallback neural often improves hard/all raw error but catastrophically damages easy cases; fallback is required for deployability. |

## Boundary

Stage42-D fresh-runs safety/floor/full-waypoint ablations and cached-verifies prior Stage30/41 component ablation evidence. It does not complete all-component retraining inside Stage42-D.

# Stage42-Y Unified Ablation Evidence

- source: `fresh_synthesis_from_stage42x_row_cache_and_retrained_ablation_reports`
- generated_at_utc: `2026-05-26T04:46:01.553602+00:00`
- git_commit: `7c0e1b6`
- input_hash: `a88056df873cad45611d380338de6e50c0248cc6dcac457cffd11d52e422f0ae`
- gate: `13 / 13`
- verdict: `stage42_y_unified_ablation_evidence_pass`

## Current Facts
- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- Stage42-Y 汇总 row-level full-waypoint cache 与 retrained ablation evidence；不是 metric 或 seconds-level 结果。
- Stage42-X 统一 cache 是本轮 row-level full-waypoint 主证据；Stage42-H 是 retrained sequence ablation；Stage42-E 是 safety-floor 研究。
- future waypoints / endpoints 只作为 train/val labels 和 eval labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test 调阈值。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Stage42-X Reference

- ADE all: `0.090014`
- ADE t50: `0.061094`
- ADE t50 seed CI low: `0.053671`
- ADE hard/failure: `0.093746`
- easy degradation: `0.001102`

## Row-Level Full-Waypoint Ablation

| ablation | source | ADE all | ADE t50 | hard | easy | loss all | loss t50 | loss hard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `floor_only` | `fresh_reference` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.090014 | 0.061094 | 0.093746 |
| `stage42j_static_expert_only` | `cached_verified_stage42r_row_cache` | 0.036222 | 0.036875 | 0.039705 | 0.000000 | 0.053792 | 0.024218 | 0.054040 |
| `stage42p_gain_harm_only` | `cached_verified_stage42r_row_cache` | 0.051537 | 0.006596 | 0.053256 | 0.008580 | 0.038477 | 0.054498 | 0.040490 |
| `stage42s_combo_no_ucy_source` | `cached_verified_stage42s_row_cache` | 0.052387 | 0.037934 | 0.054792 | 0.001102 | 0.037627 | 0.023159 | 0.038954 |
| `stage42x_unified_full` | `fresh_run_row_level_unified_cache` | 0.090014 | 0.061094 | 0.093746 | 0.001102 | 0.000000 | 0.000000 | 0.000000 |

## Retrained Sequence Ablation

| module | source | all | t50 | hard | full-minus-ablation all | t50 | hard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `history tokens` | `fresh_run` | 0.330436 | 0.325545 | 0.337275 | 0.448036 | 0.457817 | 0.470799 |
| `domain expert` | `fresh_run` | 0.736930 | 0.741477 | 0.768207 | 0.041542 | 0.041885 | 0.039867 |
| `goal/scene tokens` | `fresh_run` | 0.773151 | 0.787621 | 0.803702 | 0.005321 | -0.004259 | 0.004372 |
| `neighbor/interaction tokens` | `fresh_run` | 0.778549 | 0.782064 | 0.809417 | -0.000078 | 0.001298 | -0.001343 |

## Safety Floor Evidence

| policy | all | t50 | hard | easy degradation | switch | deployable |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `floor_only` | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | `True` |
| `ungated_endpoint_neural` | 0.296621 | 0.215203 | 0.329374 | 1.245861 | 1.000000 | `False` |
| `ungated_full_waypoint_neural` | 0.296621 | 0.215203 | 0.329374 | 1.245861 | 1.000000 | `False` |
| `teacher_raw_policy` | 0.351474 | 0.236664 | 0.350941 | 0.000000 | 0.461947 | `True` |
| `current_composite_tail_policy` | 0.210251 | 0.136522 | 0.203849 | 0.000000 | 0.341017 | `True` |

## Interpretation

- Stage42-X is now the unified row-level full-waypoint reference over ETH_UCY, TrajNet, and UCY.
- Removing the UCY full-waypoint source reverts to Stage42-S and loses t50/hard performance, so UCY source contribution is measurable.
- Retrained sequence ablation shows history tokens are the strongest proven sequence component; domain expert also contributes positively.
- Goal/scene and neighbor/interaction evidence is mixed in the current retrained sequence table and should not be overstated.
- Safety-floor evidence remains essential: ungated neural improves raw errors but is not deployable when easy degradation violates the gate.
- All claims remain raw-frame dataset-local 2.5D; Stage5C and SMC remain disabled.

<!-- STAGE42_AC_REFRESH:START -->
## Stage42-AC Latest Evidence Refresh

- source: `fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-AB is now included as auxiliary-head evidence.
- Auxiliary-head conclusion: mixed / partial; not a uniform main contribution claim.
- Stage5C and SMC remain disabled.

| item | source | status | paper use | evidence |
| --- | --- | --- | --- | --- |
| Stage42-X unified row-level full-waypoint cache | `fresh_run_from_stage42s_row_cache_and_stage42v_ucy_predictions` | `stage42_x_unified_row_level_full_waypoint_cache_pass` | main protected 2.5D full-waypoint evidence | ADE all=0.090014, t50=0.061094, hard=0.093746, easy=0.001102 |
| Stage42-Y unified ablation evidence | `fresh_synthesis_from_stage42x_row_cache_and_retrained_ablation_reports` | `stage42_y_unified_ablation_evidence_pass` | paper-table ablation synthesis | gate=13/13; history/domain positive, goal/neighbor mixed |
| Stage42-Z claim evidence audit | `fresh_audit_from_stage42_wxy_and_paper_package_artifacts` | `stage42_z_paper_claim_evidence_audit_pass` | claim boundary audit | supports protected 2.5D raw-frame paper scope; rejects metric/seconds/foundation/ungated claims |
| Stage42-AA retrained ablation matrix | `fresh_matrix_from_stage42g_rerun_plus_stage42h_i_d_z` | `stage42_aa_retrained_ablation_matrix_pass_with_jepa_transformer_boundary` | required ablation coverage matrix | gate=15/15; no-JEPA cached negative; no-Transformer proxy boundary |
| Stage42-AB auxiliary-head retrained ablation | `fresh_run` | `stage42_ab_full_waypoint_auxiliary_ablation_pass` | mixed/partial auxiliary evidence, not main uniform-positive claim | no_aux all=-0.002339, t50=-0.037443; full-minus-no-aux t50=0.005361, all=-0.008219; uniform_positive=False |

### Auxiliary-Head Delta

| delta | value |
| --- | ---: |
| `ade_all_delta_full_minus_no_aux` | -0.008219 |
| `ade_t50_delta_full_minus_no_aux` | 0.005361 |
| `ade_hard_delta_full_minus_no_aux` | -0.009027 |
| `fde_t50_delta_full_minus_no_aux` | 0.005084 |
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
