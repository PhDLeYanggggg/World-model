# Physical World Model 2.5D Results

## M3W 当前工作路线/失败/成功总账

本轮按用户要求刷新了一个单文件中文总账：

`/Users/yangyue/Downloads/World/README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`

它详细总结了 M3W 长期目标内已经尝试过的路线、失败原因、成功证据、当前 best deployable 分层、当前模型质量、仍然禁止的 claim，以及下一步最短路径。最新纳入 Stage42-ES 到 Stage42-FL：scalar proximity/occupancy 目标保留为 diagnostic，explicit source/frame/horizon group-consistency 被选为下一步 interaction/occupancy target；Stage42-EU/EV/EW/EX/EY 证明 group-risk/adaptive repair bucket 没有超过 Stage42-DI；Stage42-EZ/FA 证明 temporal/waypoint repel 分别落在 accuracy/proximity 两侧；Stage42-FB DI/FA Pareto composer 把 near@0.05 降到 1.10%，但 all/hard 各损失约 0.07pp；Stage42-FC objective-level proximity training 提升 all/t50/hard 但 near@0.05 比 DI 差约 0.48pp；Stage42-FD safety-aware teacher regularization 被 validation 选回 teacher_alpha=0 的 FC-like 控制项；Stage42-FE constrained FC/safety composer 把 FC 高精度与 DI proximity safety 组合起来：all/t50/hard `26.41% / 23.15% / 24.81%`，near@0.05 `1.32%`，比 FC 低 `0.54pp` 且不劣于 DI；Stage42-FF 冻结 FE policy 并做 exact replay + 2000-bootstrap，all/t50/t100raw/hard CI low 为 `26.08% / 22.71% / 13.46% / 24.46%`，replay diff 为 0；Stage42-FG 做 source/domain/horizon 鲁棒性审计，TrajNet robust 但 UCY 仍 weak，TrajNet|100 也有 easy-safety 弱切片，因此 broad uniform source-level claim 仍不允许；Stage42-FH 用 UCY train-only internal validation 重新选择 FE composer family，修复 UCY fallback-only 弱域，global all/t50/t100raw/hard 为 `34.98% / 28.97% / 20.57% / 33.10%`，TrajNet 和 UCY 都 positive-safe，gate `20 / 20`；Stage42-FI 冻结 FH policy 并做 exact replay + 2000-bootstrap，policy hash `f1f6e0636167fae8721a3f7195f188dcbe1a83194b04fa0625b378ad38b5aed6`，replay diff 为 `0`，CI low all/t50/t100raw/hard 为 `34.62% / 28.46% / 19.96% / 32.73%`，gate `25 / 25`；Stage42-FJ 做 frozen FH/FI source/domain/horizon/scene 鲁棒性审计，TrajNet 与 UCY 两个 domain 均 robust positive-safe，所有 powered sources 都 robust，但 TrajNet|100、UCY|50、UCY|100 仍是 horizon weak slices，gate `14 / 14`；Stage42-FK 针对弱 horizon 做 validation-only repair，global all/t50/t100raw/hard 变为 `35.18% / 28.97% / 21.13% / 33.33%`，但 weak horizons 未减少，因此 verdict 是 `pass_with_horizon_limit`；Stage42-FL 对剩余 weak horizon 做 fresh 取证，确认三个弱切片共同根因是 oracle label low-margin ambiguous，下一步应训练 horizon-specific row-level switch model。当前结论是：validation-only constrained FC→DI safety fallback 能打破 FC 的 proximity blocker，而 UCY internal-val support 使该 composer 从 TrajNet-only robust 推进到 dual-domain positive-safe；FI freeze/replay 证明该结果不是 test-tuned 偶然输出；FJ/FK/FL 允许 dual-domain/source robust claim，但禁止 uniform horizon overclaim，并解释 blocker 是低 margin/high ambiguity weak horizon。但仍然只能写 protected dataset-local/raw-frame 2.5D evidence，不能写 metric/seconds/true-3D/foundation。Stage5C 未执行，SMC 未启用。

## M3W 长期目标详细总账

本轮按用户要求新增单文件详细中文总账：

`/Users/yangyue/Downloads/World/README_M3W_GOAL_DETAILED_LEDGER_ZH.md`

它集中总结了 M3W 长期目标内做过的路线、失败原因、成功证据、当前 best deployable、当前模型质量、仍然禁止的 claim，以及最短下一步。最新纳入 Stage42-EJ/EK/EL：guarded conversion 在 legal-ready target 为 0 时保持空队列，long-objective coverage 保留 open blockers，deployment-aligned context gain-router 仍未证明 context 独立贡献。该文件是 `cached_verified` 汇总加 Stage42-EL fresh 结果，不是新训练，也不把 cached 结果写成 fresh。当前严格结论保持不变：M3W 是 protected dataset-local/raw-frame 2.5D multi-agent world-state candidate；不是 true 3D，不是 foundation，不是 metric/seconds-level；Stage5C 未执行，SMC 未启用。

## M3W 当前长期目标工作总结

本轮按用户要求新增详细中文总账：

`/Users/yangyue/Downloads/World/README_M3W_TARGET_WORK_SUMMARY_ZH.md`

它集中总结了 M3W 长期目标内已经做过的路线、失败原因、成功证据、当前 best deployable、当前模型质量，以及仍然禁止的 claim 边界。最新纳入 Stage42-DP / DQ / DR：当前 sequence/graph residual context protocol 已关闭；protected source-level group-consistency full-waypoint runtime policy 得到 fresh promotion checkpoint；paper package 已同步 post-DP/DQ evidence。当前严格结论不变：M3W 是 protected dataset-local/raw-frame 2.5D multi-agent world-state candidate；不是 true 3D，不是 foundation，不是 metric/seconds-level；Stage5C 未执行，SMC 未启用。

## M3W 长期目标结果总账 README

本轮按用户要求新增详细中文总账：

`/Users/yangyue/Downloads/World/README_M3W_GOAL_RESULTS_SUMMARY_ZH.md`

它集中回答：在 M3W 长期目标内到底做了什么、尝试了哪些路线、哪些失败了、失败原因是什么、哪些成功了、当前 best deployable 是谁、当前模型大概是什么质量，以及哪些 claim 仍然禁止。最新纳入到 Stage42-DM reviewer replay package：group-consistency runtime rows `47458`，switch exact match `True`，all/t50/t100 raw-frame/hard improvement 分别约 `+24.72% / +22.36% / +14.35% / +23.89%`，near@0.05 从 `1.94%` 降到 `1.38%`。当前严格结论不变：M3W 是 protected dataset-local/raw-frame 2.5D multi-agent world-state candidate；不是 true 3D，不是 foundation，不是 metric/seconds-level；Stage5C 未执行，SMC 未启用。

## 当前目标总账 README

最新按用户要求汇总的单文件中文总账：

`/Users/yangyue/Downloads/World/README_M3W_CURRENT_GOAL_SUMMARY_ZH.md`

它集中总结了 M3W 长期目标内已经做过的路线、失败原因、成功证据、当前 best deployable、仍然禁止的 claim 边界，以及最新 Stage42-DI/DJ/DK/DL group-consistency full-waypoint runtime policy 证据。当前严格结论不变：M3W 是 protected dataset-local/raw-frame 2.5D multi-agent world-state candidate；不是 true 3D，不是 foundation，不是 metric/seconds-level；Stage5C 未执行，SMC 未启用。

## 当前完整复盘 README

本轮按用户要求新增详细中文复盘：

`/Users/yangyue/Downloads/World/README_M3W_CURRENT_FULL_RETROSPECTIVE_ZH.md`

它集中回答：在 M3W 长期目标内做了什么、尝试了哪些路线、哪些失败及原因、哪些成功及证据、当前 best deployable 是什么，以及哪些 claim 仍然禁止。当前结论保持严格：M3W 是 protected dataset-local/raw-frame 2.5D multi-agent world-state candidate；不是 true 3D，不是 foundation，不是 metric/seconds-level；Stage5C 未执行，SMC 未启用。最新 source support closure 仍显示 ETH_UCY / TrajNet / UCY 没有完全关闭 legal/source/time/t100 blocker。

## 当前目标总复盘 README

最新单文件中文总复盘已更新：

`/Users/yangyue/Downloads/World/README_M3W_GOAL_SUMMARY_ZH.md`

它集中回答当前长期目标内做了什么、尝试了哪些路线、哪些失败及原因、哪些成功及证据、当前 best deployable 是什么，以及仍然禁止的 claim 边界。最新版本纳入 Stage42-CV/CW/CX/CY/CZ/DA/DB/DC/DD/DE/DF/DG/DH：batch runtime replay、paper refresh、evidence provenance verifier、worktree caveat classifier、paper-freeze manifest、next-action evidence queue、context rescue decision audit、context switchability / gain-harm gate、source support closure audit、full-waypoint primary deployment gap audit、all/hard/proximity repair、all/hard weighted-loss retraining、proximity/occupancy-proxy weighted retraining。结论保持严格：当前 M3W 是 protected dataset-local/raw-frame 2.5D multi-agent world-state candidate，不是 true 3D，不是 foundation，不是 metric/seconds-level，Stage5C 未执行，SMC 未启用。

当前总判定：

```text
best deployable = M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under Stage37 / teacher safety floor
dominant mechanism = baseline-family rollout context + causal history + guarded domain expert + conservative safety floor
not main claims = JEPA, Transformer, goal/scene, neighbor/interaction as independent drivers
latest audits = Stage42-CJ goal/scene gated expert, Stage42-CK neighbor/interaction gated expert, Stage42-CL paper-package context guard, Stage42-CM full-waypoint boundary audit, Stage42-CN bridge/shape composer audit, Stage42-CP composer safety/bootstrap audit, Stage42-CQ proximity-aware composer guard, and Stage42-CR proximity guard ablation
latest gate status = CJ 10 / 10, CK 11 / 11
latest paper-package refresh = Stage42-CL post-CJ/CK context guard, 11 / 11 gates
latest full-waypoint boundary audit = Stage42-CM endpoint bridge / full-waypoint shape audit, 14 / 14 gates
latest bridge/shape composer audit = Stage42-CN, 15 / 15 gates, blocker documented
latest common-validation composer = Stage42-CO, 14 / 14 gates, endpoint-vs-full rows aligned
latest composer safety/bootstrap = Stage42-CP, 14 / 14 gates, 2000-bootstrap CI and all-agent joint safety audit
latest proximity-aware composer guard = Stage42-CQ, 19 / 19 gates, near-collision caveat repaired with positive all/t50/t100/hard bootstrap evidence
latest proximity guard ablation = Stage42-CR, 19 / 19 gates, no-guard accuracy vs guarded safety Pareto boundary documented
latest batch runtime replay = Stage42-CV, 25 / 25 gates, frozen policy replay exactly matches decisions and selected_xy/ADE/FDE
latest paper refresh = Stage42-CW, runtime replay evidence propagated into paper/reproducibility/model-card package
latest provenance verifier = Stage42-CX, 21 artifacts audited, 21 gates passed
latest worktree caveat classifier = Stage42-CY, Stage42 substantive dirty files = 0
latest paper freeze manifest = Stage42-CZ, 74 files hashed, 14 / 14 gates, candidate_clean
latest next-action queue = Stage42-DA, 15 / 15 gates, top priority is legal/source support for ETH_UCY and TrajNet t100/t50 calibration
latest context rescue decision = Stage42-DB, 13 / 13 gates, decision is stop repeating current context residual/gated protocols
latest context switchability gate = Stage42-DC, 15 / 15 gates, decision is context_switchability_not_supported
latest source support closure audit = Stage42-DD, 15 / 15 gates, DA-1 remains open with explicit ETH_UCY/TrajNet/UCY blockers
latest full-waypoint deployment gap audit = Stage42-DE, 17 / 17 gates, primary deployable promotion blocked
latest all-hard/proximity repair = Stage42-DF, 12 / 14 gates, threshold/proximity repair negative vs endpoint-linear and CQ
latest all-hard weighted-loss retraining = Stage42-DG, 13 / 15 gates, reproduces Stage42-AM but does not improve it
latest proximity/occupancy weighted retraining = Stage42-DH, 15 / 16 gates, slight all gain vs AM but hard/failure primary blocker remains
latest context materiality audit = Stage42-EE, 12 / 12 gates, selected context deltas below 1pp materiality threshold
latest source terms gap audit = Stage42-EF, 13 / 13 gates, conversion_ready_now remains 0
latest paper claim refresh = Stage42-EG, 12 / 12 gates, supported claim limited to protected source-level group-consistency full-waypoint dynamics
latest source terms intake = Stage42-EH, 14 / 14 gates, fillable manual confirmation package written with conversion_ready_now still 0
latest source terms intake validator bridge = Stage42-EI, 10 / 10 gates, CG validator now reads EH intake template
latest full pytest = 786 passed in 36.07s
```

```text
Stage42-CO common validation bridge / shape composer
source = fresh_common_validation_eval_from_cached_verified_checkpoints
verdict = stage42_co_common_validation_bridge_shape_composer_pass
gates = 14 / 14
validation/test row alignment = pass
selected policy = domain_horizon_full_waypoint_composer
selected full-waypoint slices = ETH_UCY|50 and ETH_UCY|100
test vs endpoint-linear ADE:
  all = +3.02%
  t50 = +1.50%
  t100 raw diagnostic = +6.12%
  hard/failure = +3.28%
  easy degradation = +0.25%
test vs strongest floor ADE:
  all = +23.41%
  t50 = +14.95%
  t100 raw diagnostic = +19.91%
  hard/failure = +23.00%
Stage5C_executed = false
SMC_enabled = false
```

Stage42-CO resolves the Stage42-CN row-alignment blocker with fresh common validation/test evidence. Endpoint-linear bridge and full-waypoint sequence rows align exactly on validation and test. A validation-only domain/horizon composer selects full-waypoint only for ETH_UCY t50/t100; test is evaluated once. This is a real protected bridge/shape improvement over the previous endpoint-linear deployment floor, while remaining dataset-local/raw-frame 2.5D evidence, not metric/seconds-level or Stage5C/SMC evidence.

```text
Stage42-CP common validation composer safety / bootstrap
source = fresh_joint_safety_bootstrap_from_stage42_co_policy
verdict = stage42_cp_common_validation_composer_safety_pass
gates = 14 / 14
bootstrap_n = 2000
test vs endpoint-linear ADE:
  all = +3.02%, CI [+2.64%, +3.37%]
  t50 = +1.50%, CI [+0.90%, +2.09%]
  t100 raw diagnostic = +6.12%, CI [+5.39%, +6.94%]
  hard/failure = +3.28%, CI [+2.90%, +3.68%]
joint safety:
  near_collision@0.05 delta vs endpoint-linear = +0.34%
  near_collision@0.05 delta vs strongest floor = -0.05%
  jagged-rate delta vs endpoint-linear = 0.00%
Stage5C_executed = false
SMC_enabled = false
```

Stage42-CP adds statistical and all-agent joint-safety evidence to Stage42-CO. The validation-only composer has positive bootstrap lower bounds against the endpoint-linear bridge for all/t50/t100 raw-frame/hard-failure ADE. Safety remains materially guarded: near-collision@0.05 is slightly higher than endpoint-linear but still lower than the strongest floor, and smoothness does not worsen. This is protected dataset-local/raw-frame 2.5D evidence only, not metric/seconds-level, true 3D, Stage5C, or SMC evidence.

```text
Stage42-CQ proximity-aware composer guard
source = fresh_validation_selected_proximity_guard_from_stage42_co_policy
verdict = stage42_cq_proximity_aware_composer_guard_pass
gates = 19 / 19
validation-selected guard = min_sep 0.2, margin 0.005
guard input = predicted endpoint/full-waypoint rollout geometry only
test vs endpoint-linear ADE:
  all = +1.77%, CI [+1.50%, +2.05%]
  t50 = +1.07%, CI [+0.59%, +1.52%]
  t100 raw diagnostic = +3.48%, CI [+2.91%, +4.08%]
  hard/failure = +1.93%, CI [+1.63%, +2.22%]
  easy degradation = +0.25%
joint safety:
  near_collision@0.05 delta vs endpoint-linear = -0.06%
  near_collision@0.05 delta vs strongest floor = -0.45%
  jagged-rate delta vs endpoint-linear = 0.00%
Stage5C_executed = false
SMC_enabled = false
```

Stage42-CQ turns the Stage42-CP proximity caveat into a validation-selected safety policy. It gives up part of the Stage42-CO/CP accuracy gain, but it keeps all/t50/t100 raw-frame/hard-failure positive with positive bootstrap lower bounds while making near-collision@0.05 no worse than endpoint-linear or the strongest floor. This is the safer composer variant for safety-sensitive reporting; it remains protected dataset-local/raw-frame 2.5D evidence, not metric/seconds-level, true 3D, Stage5C, or SMC evidence.

```text
Stage42-CR proximity guard ablation / Pareto audit
source = fresh_synthesis_from_stage42_co_cp_cq_artifacts
verdict = stage42_cr_proximity_guard_ablation_pass
gates = 19 / 19
ablation rows:
  endpoint_linear_reference: all/t50/t100/hard = 0.00% / 0.00% / 0.00% / 0.00%, near@0.05 delta = 0.00%
  no_proximity_guard: all/t50/t100/hard = +3.02% / +1.50% / +6.12% / +3.28%, near@0.05 delta = +0.34%
  proximity_guard: all/t50/t100/hard = +1.77% / +1.07% / +3.48% / +1.93%, near@0.05 delta = -0.06%
guard contribution:
  all-ADE cost = -1.24%
  t50-ADE cost = -0.44%
  t100 raw diagnostic cost = -2.64%
  hard/failure cost = -1.35%
  near-collision@0.05 repair = -0.40%
recommendation:
  accuracy-priority diagnostic policy = no_proximity_guard
  safety-sensitive deployment policy = proximity_guard
Stage5C_executed = false
SMC_enabled = false
```

Stage42-CR turns CQ into an explicit causal/Pareto ablation. Removing the proximity guard gives higher ADE gain but worsens near-collision@0.05; adding the guard repairs the proximity caveat while preserving positive all/t50/t100 raw-frame/hard-failure gains. This is useful paper evidence for safe-switch / guard contribution, but it remains protected dataset-local/raw-frame 2.5D evidence only.

```text
Stage42-CN bridge / shape composer audit
source = fresh_synthesis_from_stage42_cm_j_x_artifacts
verdict = stage42_cn_bridge_shape_composer_audit_pass_blocker_documented
gates = 15 / 15
selected_deployment_policy = keep_endpoint_linear_bridge_floor_with_full_waypoint_auxiliary_reporting
deployable_bridge_shape_composer_available = false
common_validation_endpoint_vs_full_waypoint_comparison_available = false
full_waypoint horizon auxiliary = supported
full_waypoint all-ADE replacement = not supported
blocked next requirement = build common validation-aligned endpoint-linear-vs-full-waypoint row cache
Stage5C_executed = false
SMC_enabled = false
```

Stage42-CN turns the Stage42-CM boundary into a composer decision. It uses Stage42-J validation-only full-waypoint/static gating plus Stage42-CM/X full-waypoint artifacts. The result is honest: full-waypoint shape heads have auxiliary t50/t100 raw-frame value, but there is not yet a common validation-aligned endpoint-vs-full-waypoint row cache that would justify a new deployment switch. The deployable policy remains the endpoint-linear bridge / Stage37-teacher floor, with full-waypoint reported as auxiliary evidence.

Latest concrete Stage42 progress after the goal retrospective:

```text
Stage42-CJ validation-only goal/scene gated expert audit
source = fresh_run
verdict = stage42_cj_goal_scene_gated_expert_pass_diagnostic_no_overclaim
gates = 10 / 10
baseline_family_control all/t50/hard = 28.78% / 31.54% / 27.58%
baseline_plus_goal_scene all/t50/hard = 26.25% / 22.76% / 24.86%
baseline_plus_motion_goal_context all/t50/hard = 24.58% / 22.02% / 23.75%
selected_variant = baseline_family_control
goal_scene_rescue_success = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-CJ directly tests the Stage42-CI gap: whether a validation-only goal/scene gated expert can safely become an incremental contribution over baseline-family rollout context. It cannot under the current source-level ridge/full-waypoint protocol: both goal candidates score lower on validation and test, so the safe choice is fallback to `baseline_family_control`. This is fresh negative evidence and keeps goal/scene as mixed/diagnostic, not a main paper claim.

```text
Stage42-CK validation-only neighbor/interaction gated expert audit
source = fresh_run
verdict = stage42_ck_neighbor_interaction_gated_expert_pass_diagnostic_no_overclaim
gates = 11 / 11
baseline_family_control all/t50/hard = 28.78% / 31.54% / 27.58%
baseline_plus_scalar_neighbor all/t50/hard = 26.37% / 22.96% / 24.88%
baseline_plus_knn_graph all/t50/hard = 24.38% / 22.38% / 23.78%
baseline_plus_graph_goal all/t50/hard = 20.67% / 22.21% / 18.81%
selected_variant = baseline_family_control
neighbor_interaction_rescue_success = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-CK tests the other mixed Stage42-CI context component. It builds current-frame kNN graph features for `337991` rows with `334525` rows having neighbors, then evaluates scalar-neighbor and graph candidates under the same validation-only rule. None beat `baseline_family_control` on validation or test. This is fresh negative evidence: neighbor/interaction remains auxiliary/diagnostic, not an independent main paper claim under the current source-level ridge/full-waypoint protocol.

```text
Stage42-CL post-CJ/CK context guard paper refresh
source = fresh_synthesis_from_stage42_cj_ck_artifacts
verdict = stage42_cl_context_guard_paper_refresh_pass
gates = 11 / 11
paper files refreshed = experiment tables, ablation tables, failure taxonomy, A-journal gap
claim boundary = goal/scene and neighbor/interaction remain auxiliary/diagnostic, not independent main claims
Stage5C_executed = false
SMC_enabled = false
```

Stage42-CL propagates the CJ/CK negative context evidence into the paper package itself, so the package now explicitly blocks goal/scene and neighbor/interaction overclaims while preserving the supported main mechanism: baseline-family rollout context + causal history + guarded domain expert under a conservative safety floor.

```text
Stage42-CM endpoint bridge / full-waypoint shape audit
source = fresh_synthesis_from_stage42_full_waypoint_artifacts
verdict = stage42_cm_full_waypoint_bridge_shape_audit_pass
gates = 14 / 14
protected full-waypoint minus endpoint-linear bridge:
  all = -2.45%
  t50 = +1.15%
  t100 raw diagnostic = +8.16%
  hard/failure = -0.87%
ungated full-waypoint easy degradation = 124.59%
UCY endpoint-to-full bridge = failed blocker
Stage5C_executed = false
SMC_enabled = false
```

Stage42-CM clarifies the full-waypoint boundary. There is real protected full-waypoint evidence, especially on t50/t100 raw-frame horizons, but endpoint-linear bridge remains stronger on all-ADE. Endpoint-only success cannot be counted as learned full-waypoint world-state success, and ungated full-waypoint neural remains unsafe.

本次汇总版已吸收 Stage42-CG/CH 的最新 legal / metric-time guard：当前有 6 个 ETH/UCY source-specific calibration candidates，但 conversion_ready=0，因而 global/restricted metric-seconds claim 仍全部禁止；source terms validator 也仍为 terms_accepted=0、conversion_ready=0、converted=0、evaluated=0。

## 中文详细目标总结

Current one-file research route/failure/success summary requested by the user:

`/Users/yangyue/Downloads/World/README_M3W_LONG_GOAL_RETROSPECTIVE_ZH.md`

This is the newest detailed Chinese retrospective for the full M3W goal: what was attempted, which routes failed and why, which routes succeeded, the current best deployable model, the strict claim boundaries, and the latest Stage42-BR calibrated t50 source-support gap.

Latest concrete Stage42 progress after that summary:

```text
Stage42-BX slice-level floor relaxability audit
source = fresh_stage42_bx_floor_relaxability_audit
verdict = stage42_bx_floor_relaxability_audit_pass
gates = 14 / 14
relaxable_slices = TrajNet|25
t50_relaxable_slices = none
t50_blocked_slices = TrajNet|50, UCY|50
t100_relaxable_slices = none
teacher_floor_context_required = true
floor_free_neural_deployable = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BX refines the BW safety-floor conclusion at source/horizon level. Fallback relaxation is **not** globally deployable: only `TrajNet|25` is validation-supported and easy-safe. `TrajNet|50` is blocked by validation easy harm, all UCY slices are blocked by missing validation support in this audit, and no t100 slice is relaxable. This strengthens the deployment rule: partial fallback relaxation can be used only for validation-supported slices; teacher/floor rollout context and global safety floor remain required.

Verification: `.venv-pytorch/bin/python run_stage42_floor_relaxability_audit.py`, focused pytest `8 passed`, full pytest `507 passed in 67.02s`.

```text
Stage42-BW safety-floor necessity audit
source = fresh_stage42_bw_safety_floor_necessity_audit
verdict = stage42_bw_safety_floor_necessity_audit_pass
gates = 15 / 15
current protected all/t50/hard = 21.03% / 13.65% / 20.38%
current protected easy degradation = 0.00%
ungated endpoint/full-waypoint easy degradation = 124.59%
teacher raw policy collision/proximity warning = +1.87%
no_floor_rel_context protected t50 delta = -9.21%
no_safe_baseline_context protected t50 delta = -9.50%
small tabular neural context supported = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BW resolves the safety-floor interpretation: the Stage37/teacher floor is not just an implementation crutch. Fallback relaxation is supported for one baseline-family probe, but removing teacher/floor rollout context hurts protected t50 by about 9%, while ungated endpoint/full-waypoint variants create unacceptable easy-case harm. The current deployable path remains protected dataset-local/raw-frame 2.5D dynamics under the safety floor, with baseline-family rollout context as the dominant supported mechanism.

Verification: `.venv-pytorch/bin/python run_stage42_safety_floor_necessity_audit.py`, focused pytest `5 passed`, full pytest `502 passed in 68.35s`.

```text
Stage42-BV source acquisition / blocker matrix
source = fresh_stage42_bv_source_acquisition_status
verdict = stage42_bv_source_acquisition_status_pass_blockers_actionable
gates = 16 / 16
blockers_total = 5
blockers_active = 5
ucy_students_blocker_narrowed = true
eth_seq_blocker_resolved = false
trajnet_raw_long_source_resolved = false
global_t100_positive_claim_allowed = false
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
auto_download_executed = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BV converts the remaining external source-support problems into an actionable blocker matrix. It keeps five blockers active rather than overclaiming them: `ETH_seq_t50_source_support`, `UCY_students_t50_source_support`, `TrajNet_raw_long_t100_source_support`, `ETH_UCY_global_t100_source_support`, and `global_metric_seconds_claim`. It records official/source references for TrajNet++/AIcrowd, OpenTraj, ETH CVL, and UCY crowd data, but does not auto-download anything and does not count registry-only or terms-blocked data as converted/evaluated. Current next actions remain: provide one more independent t50-capable UCY_students source, verify ETH-Person terms, and provide legal raw long TrajNet-compatible tracks if t100 source-CV is required.

Previous concrete Stage42 progress:

```text
Stage42-BS UCY_zara family-specific t50 policy
source = fresh_ucy_zara_t50_family_policy
verdict = stage42_bs_ucy_zara_t50_family_policy_pass_positive
gates = 14 / 14
rows_total = 51544
t50_rows_total = 12750
candidate_t50_oracle_headroom_macro_mean = 0.431886
all_improvement_macro_mean = 0.061240
t50_improvement_macro_mean = 0.247189
t50_improvement_min = 0.150958
hard_failure_improvement_macro_mean = 0.067158
easy_degradation_max = 0.012388
positive_t50_fold_count = 3 / 3
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BS resolves the `UCY_zara` part of the Stage42-BR calibrated t50 blocker with a validation-only conservative switch-rate guard. It does not resolve `ETH_seq` or `UCY_students` source-support blockers, and it does not permit global metric/seconds-level claims.

Latest ETH_seq blocker follow-up:

```text
Stage42-BT ETH_seq t50 support dry-run
source = fresh_eth_seq_t50_support_dry_run_terms_unverified
verdict = stage42_bt_eth_seq_t50_support_dry_run_pass_blocker_confirmed
gates = 13 / 13
h50_windows_total = 4397
technical_h50_mean_improvement_vs_fallback = 0.411217
safe_positive_h50_fold_count = 3 / 5
eth_seq_holdout_rows = 273
eth_seq_h50_improvement_vs_fallback = 0.0
eth_seq_t50_support_repaired = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BT confirms that ETH-Person XML technical h50 signal does **not** safely repair the actual `ETH_seq_eth` calibrated t50 holdout under validation-only selection. ETH-Person terms remain unverified, and ETH_seq still needs same-family/source-compatible support or a stronger source-compatible model.

Latest UCY_students source-support audit:

```text
Stage42-BU UCY_students t50 source-support audit
source = fresh_ucy_students_t50_source_support
verdict = stage42_bu_ucy_students_t50_source_support_pass_blocker_narrowed
gates = 14 / 14
local_candidates_audited = 9
local_paths_found = 9
independent_t50_capable_sources = UCY_students01, UCY_students03
new_independent_t50_sources_found = UCY_students01
additional_independent_t50_sources_still_needed = 1
source_cv_ready = false
ucy_students_t50_support_repaired = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BU narrows the `UCY_students` blocker. The local `students001` file is a real additional t50-capable same-family source, while `students002` is locally present but too short for t50 and alternate `students003` / TrajNet / stage5b files are duplicates rather than independent sources. UCY_students therefore still lacks one independent t50-capable students-family source before train/val/holdout source-CV can be attempted.

Previous consolidated human-facing review:

`/Users/yangyue/Downloads/World/README_M3W_FULL_GOAL_REVIEW_ZH.md`

This remains a consolidated human-facing review for the full M3W goal: routes tried, failures and root causes, successful evidence, current best deployable model, current claim boundaries, and Stage42 source-level updates.

Previous canonical summary:

`/Users/yangyue/Downloads/World/README_M3W_GOAL_SUMMARY_ZH.md`

This Chinese README remains the longer ongoing ledger: what was attempted across the M3W goal, which routes failed and why, which routes succeeded, current best deployable status, strict claim boundaries, and the latest Stage42 source package updates. It keeps the key boundary explicit: current M3W is still a protected dataset-local/raw-frame 2.5D multi-agent world-state candidate, not true 3D, not foundation-scale, not metric/seconds-level, not Stage5C, and not SMC.

Companion detailed ledgers remain available:

```text
/Users/yangyue/Downloads/World/README_M3W_RESEARCH_SUMMARY_ZH.md
/Users/yangyue/Downloads/World/README_M3W_DETAILED_RESULTS_ZH.md
/Users/yangyue/Downloads/World/README_M3W_EXECUTION_SUMMARY_ZH.md
```

Latest Stage42-BD local t100 source inventory:

```text
source = fresh_local_path_inventory
verdict = stage42_bd_local_t100_source_inventory_pass
gates = 10 / 10
files_scanned = 93
parseable_files = 74
t100_capable_files = 8
already_used_t100_files = 4
novel_t100_candidate_files = 4
estimated_novel_t100_windows = 6257
stage42_be_conversion_recommended = true
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BD only identifies local conversion candidates; it does not convert, train, evaluate, or change t100 claims. The next meaningful step is Stage42-BE conversion + no-leakage + train-only source-CV for the four novel local t100 candidates.

Latest Stage42-BE local t100 conversion-readiness audit:

```text
source = fresh_local_conversion_readiness
verdict = stage42_be_local_t100_conversion_readiness_pass
gates = 12 / 12
candidate_files = 4
schema_conversion_ready_files = 4
estimated_t50_windows = 15813
estimated_t100_windows = 6257
domains_with_source_cv_after_conversion = UCY
full_feature_store_written = false
training_run = false
evaluation_run = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BE parses the four local candidates and confirms they can map to the external row schema. UCY now has enough novel local t100-capable sources for a source-CV readiness plan after actual conversion. This is still readiness evidence only; it is not a converted dataset, trained model, evaluated model, metric/seconds claim, or t100 success claim.

Latest Stage42-BF local t100 schema conversion and source-CV baseline audit:

```text
source = fresh_in_memory_schema_conversion
verdict = stage42_bf_local_t100_schema_conversion_pass
gates = 12 / 12
candidate_sources = 4
converted_sources = 4
t50_eval_windows = 15058
t100_eval_windows = 6071
source_cv_domains_evaluated = ETH_UCY, UCY
source_cv_domains_positive_vs_constant_velocity = UCY
UCY_mean_holdout_improvement_vs_constant_velocity = 0.607043
UCY_min_holdout_improvement_vs_constant_velocity = 0.491545
materialized_feature_store_written = false
training_run = false
t100_positive_claim_allowed = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BF performs an actual in-memory schema conversion and causal baseline audit. It also fixes the local UCY `obsmat_px` coordinate-layout issue by detecting the 8-column layout. UCY now has positive t100 baseline-family source-CV readiness evidence, but this is still not protected M3W policy training/evaluation; t100 remains blocked as a deployable positive claim until Stage42-BG.

Latest Stage42-BG local t100 protected policy source-CV:

```text
source = fresh_source_cv_protected_policy
verdict = stage42_bg_local_t100_protected_policy_pass_with_global_t100_blocker
gates = 13 / 13
candidate_sources = 4
t50_policy_windows = 15058
t100_policy_windows = 6071
source_cv_domains_evaluated = UCY
source_cv_domains_blocked = ETH_UCY
UCY_t100_source_cv_supported = true
UCY_t100_mean_improvement_vs_fallback = 0.440938
UCY_t100_min_improvement_vs_fallback = 0.438579
UCY_t100_max_easy_degradation = 0.011340
global_t100_positive_claim_allowed = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BG is the first follow-up after BF that selects a protected baseline-family policy on validation sources and evaluates once on held-out local sources. UCY local t100 source-CV is positive and easy-safe. This is still not a global t100 deployment claim because ETH_UCY remains under-supported and TrajNet is not represented in these new local candidates.

Latest Stage42-BH independent-source t100 audit:

```text
source = fresh_local_independent_source_audit
verdict = stage42_bh_independent_t100_source_audit_partial
gates = 13 / 14
raw_t100_capable_files = 8
independent_t100_sources = 5
duplicate_or_alternate_format_group_count = 2
UCY_independent_sources = 4
ETH_UCY_independent_sources = 1
TrajNet_independent_sources = 0
UCY_t100_mean_improvement_vs_fallback = 0.483414
UCY_t100_min_improvement_vs_fallback = 0.340559
UCY_t100_max_easy_degradation = 0.063323
global_t100_positive_claim_allowed = false
```

Stage42-BH is stricter than BG because it deduplicates alternate files from the same scene/source before source-CV. Under this stricter protocol UCY still has positive mean t100 gain, but easy degradation exceeds the 2% gate on one independent holdout, so UCY t100 support is not yet deployable. ETH_UCY and TrajNet remain hard blockers.

Latest Stage42-BI source-robust t100 easy-guard repair:

```text
source = fresh_source_robust_easy_guard_repair
verdict = stage42_bi_ucy_t100_easy_guard_repair_pass_with_global_blocker
gates = 14 / 14
UCY_independent_sources = 4
ETH_UCY_independent_sources = 1
TrajNet_independent_sources = 0
UCY_t100_mean_improvement_vs_fallback = 0.445914
UCY_t100_min_improvement_vs_fallback = 0.425313
UCY_t100_max_easy_degradation = 0.011340
BH_previous_UCY_max_easy_degradation = 0.063323
global_t100_positive_claim_allowed = false
```

Stage42-BI repairs the UCY independent-source t100 easy-degradation blocker by requiring a candidate policy to be positive and easy-safe on every non-holdout source before holdout evaluation. This is a real fresh repair for UCY t100, but global t100 is still blocked because ETH_UCY and TrajNet lack enough independent t100 sources.

Latest Stage42-BJ post-BI t100 source package:

```text
source = fresh_post_bi_t100_source_package
verdict = stage42_bj_post_bi_t100_source_package_pass
gates = 14 / 14
UCY_t100_repaired = true
ETH_UCY_independent_sources = 1
ETH_UCY_additional_sources_needed = 2
TrajNet_independent_sources = 0
TrajNet_additional_sources_needed = 3
local_inventory_exhausted_for_domains = ETH_UCY, TrajNet
global_t100_positive_claim_allowed = false
auto_download_executed = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BJ turns the BI blocker into an explicit acquisition / user-action package. It distinguishes raw t100-capable files from independent scene/source groups: the local ETH_UCY files collapse to one independent `ETH/seq_eth` source, and TrajNet has no independent local t100 source. Therefore the next real progress requires legal official/user-provided independent t100 sources for ETH_UCY and TrajNet, followed by conversion and train-only source-CV. No data was auto-downloaded and no registry-only source was counted as converted/evaluated.

Latest Stage42-BK post-BJ local source verification:

```text
source = fresh_post_bj_local_source_verification
verdict = stage42_bk_local_source_verification_pass
gates = 11 / 11
ETH_UCY parsed files = 18
ETH_UCY t100-capable files = 7
ETH_UCY independent t100 groups = 6
ETH_UCY potential new groups vs BJ = 5
ETH-Person XML t100 candidates = 5
TrajNet parsed files = 59
TrajNet t100-capable files = 0
TrajNet loader gap = fixed short snippets, not raw long tracks
global_t100_positive_claim_allowed = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BK closes a real loader-audit gap: local `ETH-Person/data/*.xml` files contain t100-capable ETH_UCY-style tracks and can potentially repair ETH_UCY source support after license/terms confirmation, conversion, no-leakage, and train-only source-CV. It does not count these XML files as converted/evaluated yet. For TrajNet, the local files parse as 8/20-step challenge snippets, so they cannot repair raw-frame t100; longer official/user-provided raw sources are still required.

Latest Stage42-BL ETH-Person XML technical conversion dry-run:

```text
source = fresh_technical_dry_run_terms_unverified
verdict = stage42_bl_eth_person_xml_t100_dry_run_pass
gates = 13 / 13
candidate_sources = 5
strict_independent_sources = 5
eth_person_xml_sources = 4
t100_windows_total = 1485
source_cv_folds = 5
technical_t100_all_folds_safe_positive = true
technical_t100_mean_improvement_vs_fallback = 0.683549
technical_t100_min_improvement_vs_fallback = 0.496424
technical_t100_max_easy_degradation = -0.014155
license_terms_confirmed = false
official_converted_dataset_claim_allowed = false
deployable_t100_claim_allowed = false
global_t100_positive_claim_allowed = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BL proves the ETH-Person XML loader and strict source-CV pipeline are technically executable and strongly positive in a dry-run. It still does **not** change the official claim boundary: ETH-Person terms are unconfirmed, the result is not counted as official converted/evaluated data, and global/deployable t100 remains blocked until user confirmation plus official conversion/no-leakage/source-CV rerun.

Latest Stage42-BM ETH-Person terms / official-use audit:

```text
source = fresh_eth_person_terms_audit
verdict = stage42_bm_eth_person_terms_audit_pass_claim_blocked
gates = 14 / 14
BL_technical_t100_all_folds_safe_positive = true
BL_technical_t100_mean_improvement_vs_fallback = 0.683549
BL_technical_t100_min_improvement_vs_fallback = 0.496424
BL_technical_t100_max_easy_degradation = -0.014155
OpenTraj_toolkit_license_can_cover_ETH_Person_dataset = false
ETH_Person_local_terms_found = false
official_source_url_recorded = true
official_terms_verified = false
official_converted_dataset_claim_allowed = false
deployable_t100_claim_allowed = false
global_t100_positive_claim_allowed = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BM keeps the positive BL dry-run as technical evidence but blocks any official/deployable/global t100 claim. The OpenTraj root MIT license is treated as toolkit/software license only, not permission for ETH-Person data. The user-action file is `outputs/stage42_long_research/user_action_required_eth_person_terms_stage42.md`; ETH-Person XML can only advance after official terms or user-confirmed permission are provided.

Latest Stage42-BN strict source time/geometry calibration audit:

```text
source = fresh_source_time_geometry_calibration_audit
verdict = stage42_bn_source_time_geometry_calibration_pass_with_global_claim_blocked
gates = 13 / 13
source_records_audited = 7
ETH source-specific metric/time sources = 2
UCY source-specific metric/time sources = 4
source-specific calibrated candidates =
  ETH_seq_eth, ETH_seq_hotel,
  UCY_zara01, UCY_zara02, UCY_zara03, UCY_students03
SDD scale_count = 60
SDD metric claim allowed = false
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
M3W official metric/seconds claim allowed = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BN moves calibration forward without overclaiming. ETH `seq_eth` / `seq_hotel` and several UCY sources have local evidence for source-specific annotation-step geometry/time calibration, but this does not upgrade the global M3W claim. Current reports must still use raw-frame / dataset-local wording unless a future evaluation explicitly restricts itself to a verified source-specific calibrated subset.

Latest Stage42-BO calibrated-subset source-CV evaluation:

```text
source = fresh_calibrated_subset_source_cv
verdict = stage42_bo_calibrated_subset_eval_partial
gates = 10 / 13
calibrated_sources_evaluated = 6
rows_total = 160338
all_improvement_macro_mean = 0.090510
all_improvement_min = 0.0
t50_improvement_macro_mean = 0.070729
t50_improvement_min = -0.107784
t100_raw_frame_diagnostic_macro_mean = 0.104071
hard_failure_improvement_macro_mean = 0.097944
easy_degradation_max = 1.032550
positive_all_folds = false
positive_t50_folds = false
easy_safe_all_folds = false
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BO is the first fresh source-CV evaluation restricted to the Stage42-BN calibrated candidates. It shows real positive macro signal, but it is **not deployable** as a source-calibrated policy: `UCY_students03` has severe easy harm and `ETH_seq_eth` has negative t50. This is why BO is recorded as partial evidence and not a calibrated-subset success.

Latest Stage42-BP calibrated-subset safety repair:

```text
source = fresh_calibrated_subset_safety_repair
verdict = stage42_bp_calibrated_subset_safety_repair_pass_limited_positive
gates = 11 / 11
source_cv_folds = 6
rows_total = 160338
all_improvement_macro_mean = 0.057580
all_improvement_min = 0.0
t50_improvement_macro_mean = 0.061868
t50_improvement_min = -0.066609
t100_raw_frame_diagnostic_macro_mean = 0.028177
hard_failure_improvement_macro_mean = 0.056282
easy_degradation_max = 0.0
positive_fold_count = 3
positive_t50_fold_count = 2
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BP adds train+val source/source-family support guards. It fixes BO's worst over-switching: `UCY_students03` is now guarded to fallback instead of harming easy cases. This is a **limited positive safety repair**, not global metric/seconds success and not uniform t50 success, because `ETH_seq_eth` t50 remains negative and several sources remain fallback-only.

Verification: `python3 run_stage42_calibrated_subset_eval.py` passed with partial BO evidence, `python3 run_stage42_calibrated_subset_safety_repair.py` passed with `11 / 11` BP gates, focused pytest passed with `11` tests, and `python3 -m pytest tests` passed with `484` tests.

Latest Stage42-BQ calibrated-subset t50 source-family support repair:

```text
source = fresh_calibrated_subset_t50_support_repair
verdict = stage42_bq_calibrated_subset_t50_support_repair_pass_t50_nonharm_limited_positive
gates = 12 / 12
source_cv_folds = 6
rows_total = 160338
t50_min_source_family_support = 2
all_improvement_macro_mean = 0.042380
all_improvement_min = 0.0
t50_improvement_macro_mean = 0.0
t50_improvement_min = 0.0
t100_raw_frame_diagnostic_macro_mean = 0.027796
hard_failure_improvement_macro_mean = 0.040266
easy_degradation_max = 0.0
positive_fold_count = 3
positive_t50_fold_count = 0
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BQ addresses BP's remaining `ETH_seq_eth` t50 negative slice by requiring at least two independent train+val sources from the same source-family before any t50 switch is allowed. This repairs t50 **non-harm**: the minimum t50 improvement becomes `0.0` and easy remains safe. It does not prove positive t50 transfer on calibrated subsets because every t50 fold is now fallback/non-positive; the correct claim is limited positive all/hard evidence with t50 guarded to the floor.

Verification: `python3 run_stage42_calibrated_subset_t50_support_repair.py` passed with `12 / 12` BQ gates, focused pytest passed with `10` BO/BP/BQ tests, and `python3 -m pytest tests` passed with `487` tests.

Latest Stage42-BR calibrated t50 source-support gap audit:

```text
source = fresh_calibrated_t50_source_support_gap_audit
verdict = stage42_br_calibrated_t50_source_support_gap_audit_pass
gates = 12 / 12
families_audited = 3
calibrated_sources_audited = 6
unsupported_family_holdout_count = 3
families_with_additional_sources_needed = ETH_seq, UCY_students
families_with_support_but_no_positive_t50 = UCY_zara
BQ_t50_macro = 0.0
BQ_t50_min = 0.0
BQ_positive_t50_fold_count = 0
ETH-Person XML local candidates = 5
ETH-Person terms verified = false
TrajNet local t100-capable files = 0
positive_t50_claim_allowed = false
t50_nonharm_claim_allowed = true
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BR turns the BQ non-harm result into a concrete blocker/action list. `ETH_seq` needs one more same-family calibrated source and has five local ETH-Person XML candidates, but they remain blocked by unverified terms. `UCY_students` needs two additional same-family calibrated sources. `UCY_zara` already has enough source-family support, so its failure is policy/model-side: the validation-safe t50 policy still falls back to the floor. This is gap evidence and user-action evidence, not positive calibrated t50 transfer.

Verification: `python3 run_stage42_calibrated_t50_source_support_gap_audit.py` passed with `12 / 12` BR gates, focused pytest passed with `6` BQ/BR tests, and `python3 -m pytest tests` passed with `490` tests.

Previous long-form research ledger:

`/Users/yangyue/Downloads/World/README_M3W_RESEARCH_SUMMARY_ZH.md`

Latest update: this canonical Chinese summary now explicitly includes Stage42-W/X/Y/Z/AA/AB/AC plus Stage42-AD/AE/AF/AG/AH/AI/AJ/AK/AL/AM/AN/AO/AP/AQ/AR/AS/AT/AU/AV/AW/AX/AY/AZ/BA/BB/BC/BD/BE evidence refreshes and a user-requested detailed route review: what was attempted, what failed, why it failed, what worked, current best deployable status, full-waypoint auxiliary-head mixed evidence, weak-slice/source/easy-safety repairs, t100 data/source-support blocker, local t100 source inventory/readiness, and the no-true-3D/no-metric/no-seconds/no-Stage5C/no-SMC claim constraints.

Latest direct user-facing summary refresh: `/Users/yangyue/Downloads/World/README_M3W_RESEARCH_SUMMARY_ZH.md` now starts with a compact but detailed “本次交付版总摘要”. It summarizes the routes tried, main failure modes, successful stages, current best deployable model, claim boundaries, and next shortest path. This is a documentation-only refresh based on cached verified reports and does not re-label any `not_run` or failed branch as successful.

Validation for the latest detailed summary / Stage42-BJ refresh: focused pytest `5 passed`; full-suite pytest remains deferred for the ongoing long-running Stage42 goal.

Most important current summary:

```text
current best deployable =
  M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
  under Stage37 / teacher safety floor

honest claim =
  protected dataset-local raw-frame 2.5D multi-agent world-state candidate

not allowed =
  true 3D / foundation / metric / seconds-level / ungated neural / Stage5C / SMC
```

The updated summary also records the main failed routes:

```text
JEPA downstream lift = not proven
hard-class selector = failed due to margin ambiguity and easy harm
SDD -> external zero-shot = failed due to coordinate/goal/horizon/domain gap
ordinary residual/correction = not deployable
ungated neural dynamics = unsafe
endpoint-to-full UCY bridge = failed
auxiliary heads = mixed/partial, not a uniform main claim
```

Latest Stage42-AZ AY shadow-holdout t100 robustness audit:

```text
source = fresh_run
verdict = stage42_az_shadow_holdout_robustness_pass_with_ay_t100_limitation
gates = 16 / 16
AY_strict_guard_shadow_h100_easy_degradation = 0.122946
source_support_guard_all = 0.133351
source_support_guard_t50 = 0.121766
source_support_guard_t100_raw_frame_diagnostic = 0.000000
source_support_guard_hard_failure = 0.127756
source_support_guard_easy_degradation = -0.022205
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AZ is negative/repair evidence, not a new t100 success. It builds shadow train/val/holdout only from original train sources and excludes final val/test from thresholding. The AY strict t100 guard is not independently robust on shadow holdout because ETH_UCY t100 easy harm appears. A stricter source-support guard protects easy and keeps all/t50/hard positive, but it removes positive t100 gain on this shadow holdout. The correct claim is therefore: t100 remains raw-frame diagnostic and needs more independent validation support; do not write it as a stable long-horizon success.

Latest Stage42-BA train-only t100 source-CV repair:

```text
source = fresh_run
verdict = stage42_ba_t100_source_cv_repair_pass_with_t100_blocker
gates = 16 / 16
source_cv_folds = 7
ETH_UCY_safe_positive_t100_folds = 0 / 4
TrajNet_safe_positive_t100_folds = 1 / 3
UCY_status = not_run_fewer_than_three_t100_capable_original_train_sources
after_cv_guard_all = 0.280997
after_cv_guard_t50 = 0.289698
after_cv_guard_t100_raw_frame_diagnostic = 0.000000
after_cv_guard_hard_failure = 0.251576
after_cv_guard_easy_degradation = -0.372431
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BA takes the AZ limitation seriously instead of trying to polish it away. It uses only original train sources to run t100 source-CV and requires at least two safe-positive source folds before any domain keeps a t100 slice. No domain passes that support rule, so all domain|100 slices are guarded to the causal floor before final test evaluation. The final protected result still has strong all/t50/hard improvement and safe easy cases, but t100 is zeroed out. That is the honest deployment boundary: t100 is a blocker/diagnostic, not a current positive-transfer claim.

Latest Stage42-AM proposed source-level full-waypoint evaluation:

```text
source = fresh_run
verdict = stage42_am_source_level_full_waypoint_eval_pass_positive
gates = 12 / 12
proposed_source_level_test_rows = 47458
TrajNet_test_rows = 37918
UCY_test_rows = 9540
test_full_waypoint_rows = 32056
ADE_all = 0.245788
ADE_t50 = 0.220171
ADE_t100_raw_frame_diagnostic = 0.143652
ADE_hard_failure = 0.237494
easy_degradation = -0.256627
ADE_all_CI_low = 0.242554
ADE_t50_CI_low = 0.215923
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AM addresses the Stage42-AL coverage gap by evaluating the proposed source-level test split directly rather than reusing the locked-policy stress pool. It trains a past-only ridge full-waypoint probe on proposed train rows, selects the safety policy on validation only, and evaluates test once. It is positive source-level raw-frame full-waypoint evidence, but it is still a protected dataset-local 2.5D probe, not metric/seconds-level, true-3D, foundation, Stage5C, or SMC evidence.

Latest Stage42-AN proposed source-level retrained ablation:

```text
source = fresh_run
verdict = stage42_an_source_level_ablation_partial_component_evidence
gates = 9 / 10
full_ADE_all = 0.245788
full_ADE_t50 = 0.220171
full_ADE_hard_failure = 0.237494
positive_independent_components = baseline_family_context
positive_combined_variants = motion_goal_no_baseline_domain
not_proven_independent = history, neighbor_interaction, goal_prototype, domain_expert, safe_switch necessity in this ridge probe
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AN is an important boundary result, not a clean success. Each ablation is retrained and validation-selected on the proposed source-level split. It confirms the full ridge probe remains strong, but it does **not** prove two independent module contributions: only baseline/family context is independently supported. History, neighbor, goal prototype, domain expert, and safe-switch necessity need stronger neural/graph ablation or richer source-level features before they can be paper main claims.

Latest Stage42-AO proposed source-level incremental / standalone retrained ablation:

```text
source = fresh_run
verdict = stage42_ao_incremental_component_evidence_partial_or_negative
gates = 10 / 11
full_ADE_all = 0.245788
full_ADE_t50 = 0.220171
baseline_family_only_ADE_all = 0.287773
baseline_family_only_ADE_t50 = 0.315425
positive_standalone_context_variants = history_only, motion_goal_context
positive_incremental_context_variants = none
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AO answers the next question after Stage42-AN: whether history / goal / neighbor context has standalone or incremental value after baseline-family rollout context. The result is a useful boundary/negative finding. `history_only` and `motion_goal_context` show standalone positive signal, but no context variant improves over `baseline_family_only` by the meaningful threshold; in fact `baseline_family_only` is stronger than the full ridge variant on all/t50/hard. This means the current proposed source-level ridge evidence is dominated by baseline-family rollout context. It does not prove history/goal/neighbor are useless, but it does mean they cannot yet be written as independent paper main claims without a stronger neural/graph retraining protocol or richer source-level context.

Latest Stage42-AP proposed source-level residual-context retraining:

```text
source = fresh_run
verdict = stage42_ap_residual_context_evidence_partial_or_negative
gates = 8 / 9
baseline_family_only_ADE_all = 0.287773
baseline_family_only_ADE_t50 = 0.315425
baseline_family_only_ADE_hard_failure = 0.275812
positive_residual_context_variants = none
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AP tests whether history / goal / neighbor context can explain residual errors left by the baseline-family first-stage model. It trains baseline-family first, then trains residual context variants with validation-selected residual alpha and safety policy. The result is another useful boundary: residual history/goal/neighbor variants only change all/t50/hard by tiny amounts and none clears the +1% residual-increment threshold. This strengthens the Stage42-AO conclusion: under the current proposed source-level ridge protocol, context modules are not independently supported beyond baseline-family rollout context. The next real fix must be a stronger neural/graph context protocol or richer context features, not more ridge context claim polishing.

Latest Stage42-AQ proposed source-level neural residual-context retraining:

```text
source = fresh_run
runtime = .venv-pytorch/bin/python arm64, torch_threads=4, num_workers=0
verdict = stage42_aq_neural_context_evidence_partial_or_negative
gates = 11 / 12
baseline_family_only_ADE_all = 0.287773
baseline_family_only_ADE_t50 = 0.315425
baseline_family_only_ADE_hard_failure = 0.275812
positive_neural_context_variants = none
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AQ replaces the residual second stage with a real PyTorch MLP, trained in the arm64 `.venv-pytorch` runtime, to test whether tabular neural history/goal/neighbor context can learn residual full-waypoint dynamics beyond baseline-family rollout context. It still fails the increment gate: `neural_history`, `neural_goal_neighbor`, and `neural_history_goal_neighbor` all underperform the baseline-family first-stage on all/t50/hard. This rules out a simple tabular neural-context fix. The next meaningful direction is graph/sequence/scene-rich context, not another tabular MLP.

Latest Stage42-AR proposed source-level sequence-context residual training:

```text
source = fresh_run
runtime = .venv-pytorch/bin/python arm64, torch_threads=4, num_workers=0
verdict = stage42_ar_sequence_context_evidence_partial_or_negative
gates = 11 / 12
history_seq_shape = [337991, 64, 7]
baseline_family_only_ADE_all = 0.287773
baseline_family_only_ADE_t50 = 0.315425
baseline_family_only_ADE_hard_failure = 0.275812
positive_sequence_context_variants = none
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AR upgrades AQ from tabular MLP to a temporal Conv1D sequence encoder over past-only `history_seq` plus goal/neighbor context. It still fails the increment gate: `sequence_history`, `sequence_goal_neighbor_no_history`, and `sequence_history_goal_neighbor` all underperform the baseline-family first-stage on all/t50/hard. This is now a strong boundary result: current source-level success is not explained by independent history/goal/neighbor residual modules under ridge, tabular MLP, or temporal Conv1D residual protocols. The next credible experiment must use richer graph/scene tokens, a different supervision target, or explicitly evaluate whether baseline-family rollout context itself should be the main paper contribution.

Latest Stage42-AS proposed source-level graph-interaction context residual training:

```text
source = fresh_run
verdict = stage42_as_graph_context_evidence_partial_or_negative
gates = 10 / 11
rows_with_neighbors = 334525
max_unique_agents_per_frame = 65
baseline_family_only_ADE_all = 0.287773
baseline_family_only_ADE_t50 = 0.315425
baseline_family_only_ADE_hard_failure = 0.275812
graph_only_ADE_all = 0.264270
graph_goal_ADE_all = 0.264651
graph_history_goal_ADE_all = 0.264765
positive_graph_context_variants = none
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AS builds structured current-frame kNN graph / interaction features from `(source_file, frame_id)`, deduplicates horizon rows by agent, excludes the target agent, and uses only current/past motion. It still fails the increment gate: graph-only, graph+goal, and graph+history+goal residual variants all underperform the baseline-family first stage on all/t50/hard. This further narrows the claim boundary: the current proposed source-level evidence is dominated by baseline-family rollout context, not by independent history/goal/neighbor/sequence/graph residual modules under the tested protocols.

Latest Stage42-AT proposed source-level safety-floor / fallback audit:

```text
source = fresh_run
verdict = stage42_at_source_level_fallback_audit_pass
gates = 11 / 11
baseline_family_protected_all = 0.287773
baseline_family_protected_t50 = 0.315425
baseline_family_ungated_all = 0.461656
baseline_family_ungated_t50 = 0.411874
baseline_family_ungated_hard_failure = 0.458447
baseline_family_ungated_easy_degradation = -0.305625
fallback_removal_for_baseline_family_probe = supported_on_this_source_level_split
teacher_floor_context_removal = not_supported_as_global_replacement
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AT separates two things that should not be conflated. Removing the fallback switch for the source-level baseline-family ridge probe is safe and stronger on this split. Removing teacher/floor rollout context as a global replacement is not supported: floor/safe context removals hurt protected t50 and do not justify a floor-free neural claim. The correct claim is therefore narrower but useful: source-level baseline-family rollout context is strong enough that fallback can be relaxed in this probe; it is not evidence that unrestricted neural dynamics can replace the Stage37/teacher floor.

Latest Stage42-AU proposed source-level baseline-family mechanism audit:

```text
source = fresh_run
verdict = stage42_au_baseline_family_mechanism_pass
gates = 11 / 11
horizon_domain_control_protected_all = 0.000000
floor_rel_only_protected_all = 0.036215
safe_baseline_rel_only_protected_t50 = -0.099422
family_baseline_rel_only_protected_all = 0.273815
family_baseline_rel_only_protected_t50 = 0.237296
baseline_family_all_protected_all = 0.287773
baseline_family_all_protected_t50 = 0.315425
protected_multi_family_increment_supported = true
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AU explains the mechanism behind the recent source-level positives. The result is not just horizon/domain control, and not a single floor-relative feature. The strongest single component is `family_baseline_rel_only`, while `baseline_family_all` improves protected t50 further. The paper claim should therefore foreground baseline-family rollout context as the current supported mechanism, while keeping the boundary that history/goal/neighbor/sequence/graph context remains unproven as an independent contribution under the tested protocols.

Latest Stage42-AV baseline-family robustness / weak-slice audit:

```text
source = cached_verified_from_stage42_au
verdict = stage42_av_baseline_family_robustness_pass_with_limits
gates = 12 / 12
baseline_family_all all CI low = 0.284243
baseline_family_all t50 CI low = 0.309806
baseline_family_all hard/failure CI low = 0.271961
baseline_family_all easy degradation CI high = -0.459376
positive_domains = TrajNet
floor_only_or_blocked_domains = UCY
weak_horizons = 100
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AV prevents overclaiming the AU result. The global bootstrap evidence is strong and TrajNet is positive, but UCY has no validation rows in this proposed source-level split and therefore remains floor-only. Horizon 100 is still raw-frame diagnostic and has an easy-safety weak slice. The allowed claim is global / TrajNet source-level baseline-family mechanism evidence with explicit UCY and t100 limitations, not uniform domain/horizon success.

Latest Stage42-AW UCY validation-support repair:

```text
source = fresh_run
verdict = stage42_aw_ucy_validation_support_repair_pass
gates = 14 / 14
internal_val_group = UCY::UCY/zara03/crowds_zara03.txt
original_UCY_val_rows = 0
repaired_UCY_val_rows = 9540
validation_best_variant = family_baseline_rel_only
global_all = 0.356806
global_t50 = 0.289698
global_hard_failure = 0.338904
UCY_all = 0.374492
UCY_t50 = 0.245320
UCY_hard_failure = 0.355073
UCY_easy_degradation = -0.418376
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AW repairs the UCY floor-only blocker by carving an internal validation source from UCY train sources only. Test sources remain unchanged, and thresholds/policies are selected without test metrics. This supports UCY positive transfer under a repaired validation-support protocol, while preserving the raw-frame/dataset-local claim boundary.

Latest Stage42-AX repaired validation-support protocol robustness audit:

```text
source = cached_verified_from_stage42_aw
verdict = stage42_ax_repaired_protocol_robustness_pass_with_t100_limit
gates = 14 / 14
global_all_CI_low = 0.353076
global_t50_CI_low = 0.285398
global_t100_raw_frame_diagnostic_CI_low = 0.202944
global_hard_failure_CI_low = 0.335229
global_easy_degradation_CI_high = -0.566748
positive_domains = TrajNet, UCY
weak_horizons = 100
horizon100_easy_degradation = 0.023961
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AX verifies that the AW repaired protocol has strong global bootstrap support and positive TrajNet/UCY source-level evidence. It also keeps the limitation explicit: horizon 100 remains raw-frame diagnostic and has an easy-safety weak slice, so uniform horizon success, metric, seconds-level, true-3D, Stage5C, and SMC claims remain disallowed.

Validation for Stage42-AX:

```text
python3 run_stage42_repaired_protocol_robustness.py
  -> stage42_ax_repaired_protocol_robustness_pass_with_t100_limit (14/14)
python3 -m pytest tests/test_stage42_repaired_protocol_robustness.py
  -> 4 passed
python3 -m pytest tests
  -> 410 passed
```

Latest Stage42-AY AW repaired-protocol t100 easy-safety repair:

```text
source = fresh_run
verdict = stage42_ay_t100_easy_safety_repair_pass
gates = 17 / 17
validation_best_variant = family_baseline_rel_only
guarded_slice = TrajNet|100
kept_t100_slices = ETH_UCY|100, UCY|100
h100_easy_before = 0.023961
h100_easy_after = -0.006504
h100_easy_CI_high = 0.009833
global_all_after = 0.305467
global_t50_after = 0.289698
global_t100_raw_frame_diagnostic_after = 0.067836
global_hard_failure_after = 0.279764
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AY repairs the Stage42-AX horizon=100 easy-safety weak slice with a strict validation-only t100 guard. It sacrifices some t100 diagnostic gain for safety: h100 t100 raw-frame diagnostic drops from +21.02% to +6.78%, while h100 easy degradation improves from +2.396% to -0.650% and h100 easy CI high is below 2%. This is a safer repaired-policy candidate, not metric/seconds-level evidence and not an independent future holdout confirmation.

Validation for Stage42-AY:

```text
python3 run_stage42_aw_t100_easy_safety_repair.py
  -> stage42_ay_t100_easy_safety_repair_pass (17/17)
python3 -m pytest tests/test_stage42_aw_t100_easy_safety_repair.py
  -> 4 passed
python3 -m pytest tests
  -> 414 passed
```

Latest Stage42-AZ AY shadow-holdout t100 robustness audit:

```text
source = fresh_run
verdict = stage42_az_shadow_holdout_robustness_pass_with_ay_t100_limitation
gates = 16 / 16
AY_strict_guard_shadow_h100_easy_degradation = 0.122946
source_support_guard_all = 0.133351
source_support_guard_t50 = 0.121766
source_support_guard_t100_raw_frame_diagnostic = 0.000000
source_support_guard_hard_failure = 0.127756
source_support_guard_easy_degradation = -0.022205
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AZ builds a shadow split only from original train sources and excludes final val/test from policy selection. It shows that the Stage42-AY strict guard is not independently robust for t100 easy safety on the shadow holdout: ETH_UCY t100 easy harm appears. A stricter source-support guard protects easy and keeps all/t50/hard positive, but removes positive t100 gain. This is a claim-boundary repair: t100 remains raw-frame diagnostic and should not be written as a stable seconds-level or uniformly robust long-horizon success.

Validation for Stage42-AZ:

```text
python3 run_stage42_ay_shadow_holdout_robustness.py
  -> stage42_az_shadow_holdout_robustness_pass_with_ay_t100_limitation (16/16)
python3 -m pytest tests/test_stage42_ay_shadow_holdout_robustness.py
  -> 4 passed
python3 -m pytest tests
  -> 418 passed
```

Latest Stage42-BA train-only t100 source-CV repair:

```text
source = fresh_run
verdict = stage42_ba_t100_source_cv_repair_pass_with_t100_blocker
gates = 16 / 16
source_cv_folds = 7
ETH_UCY_safe_positive_t100_folds = 0 / 4
TrajNet_safe_positive_t100_folds = 1 / 3
UCY_status = not_run_fewer_than_three_t100_capable_original_train_sources
after_cv_guard_all = 0.280997
after_cv_guard_t50 = 0.289698
after_cv_guard_t100_raw_frame_diagnostic = 0.000000
after_cv_guard_hard_failure = 0.251576
after_cv_guard_easy_degradation = -0.372431
Stage5C_executed = false
SMC_enabled = false
```

Stage42-BA uses train-only source-CV to decide whether any domain has enough independent t100 support. ETH_UCY and TrajNet both fail the safe-positive fold rule, and UCY has too few t100-capable train sources for this audit. The final source-CV guard keeps all/t50/hard positive and easy safe, but guards all t100 slices to the causal floor. This is a blocker result for t100 positive claims, not a failure of the all/t50/hard deployable evidence.

Validation for Stage42-BA:

```text
python3 run_stage42_t100_source_cv_repair.py
  -> stage42_ba_t100_source_cv_repair_pass_with_t100_blocker (16/16)
python3 -m pytest tests/test_stage42_t100_source_cv_repair.py
  -> 4 passed
python3 -m pytest tests
  -> 422 passed
```

Latest Stage42-AD calibration evidence refresh:

```text
source = fresh_run
verdict = stage42_ad_calibration_evidence_refresh_pass
gates = 10 / 10
datasets_audited = 7
evidence_files_scanned = 1152
parseable_homography_like = OpenTraj, ETH/UCY, UCY
fps_evidence = SDD, OpenTraj, ETH/UCY, UCY
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
traffic_metric_diagnostic_only = TGSIM
```

Stage42-AD separates “calibration-like evidence exists” from “metric/seconds claim allowed”. ETH/UCY and UCY have parseable homography-like files and some FPS/stride evidence, but official pedestrian metric/seconds claims remain blocked until source-specific homography direction, coordinate convention, annotation stride, FPS, and scale are manually verified.

Latest Stage42-AE unified row-cache stress audit:

```text
source = fresh_synthesis_from_stage42x_row_level_cache
verdict = stage42_ae_unified_row_cache_stress_pass_with_limitations
gates = 12 / 12
Stage42-X reference ADE all = 0.0900
Stage42-X reference ADE t50 = 0.0611
Stage42-X t50 seed CI low = 0.0537
strong_domains = ETH_UCY, TrajNet, UCY
weak_domain = ETH_UCY for t50/FDE@50 lower bounds
weak_horizon = 25
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AE strengthens the Stage42-X paper evidence by identifying both stable and weak slices. Global t50 remains positive, but claims must not be written as uniformly positive across every domain/horizon/FDE slice.

Latest Stage42-AF weak-slice validation-margin guard repair:

```text
source = fresh_run_from_stage42x_cache_and_stage42r_validation_margin
verdict = stage42_af_weak_slice_guard_repair_pass_with_eth_t50_limitation
gates = 13 / 13
guard_threshold = validation score < 0.02
uses_test_metrics_for_threshold = false
horizon25_ADE_before = -0.004781
horizon25_ADE_after = 0.000000
ADE_all = 0.090682
ADE_t50 = 0.061094
ADE_t50_CI_low = 0.053671
ADE_hard_failure = 0.094649
easy_degradation_CI_high = 0.006233
ETH_UCY_t50_limitation_remaining = true
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AF repairs the Stage42-AE horizon=25 weak slice with a validation-only low-margin guard that forces low-validation-margin non-UCY domain/horizon choices back to the safety floor. It is a real safety repair, but not a universal fix: ETH_UCY t50/FDE@50 lower-bound weakness remains and must stay in the limitations.

Latest Stage42-AG ETH_UCY t50/FDE validation-only source repair:

```text
source = fresh_run_from_stage42x_stage42r_stage42af_validation_fde_repair
verdict = stage42_ag_eth_t50_fde_source_repair_pass
gates = 13 / 13
target_slice = ETH_UCY|50
validation_FDE@50_threshold = 0.05
uses_test_metrics_for_threshold = false
ETH_UCY_t50_ADE_CI_low_before = -0.013218
ETH_UCY_t50_ADE_CI_low_after = 0.002821
ETH_UCY_FDE@50_CI_low_before = -0.041990
ETH_UCY_FDE@50_CI_low_after = 0.021040
ADE_all = 0.091656
ADE_t50 = 0.064957
ADE_t50_CI_low = 0.058513
ADE_hard_failure = 0.095716
easy_degradation_CI_high = 0.003348
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AG addresses the remaining Stage42-AF ETH_UCY t50/FDE@50 limitation. It promotes the static expert source on `ETH_UCY|50` only when validation FDE@50 support is strong and otherwise falls back to the safety floor. This repairs the ETH_UCY t50/FDE@50 lower bounds without test threshold tuning, while claims remain dataset-local raw-frame 2.5D.

Latest Stage42-AH post-repair stress / paper-claim refresh:

```text
source = fresh_synthesis_from_stage42ag_post_repair_stress
verdict = stage42_ah_post_repair_claim_refresh_pass
gates = 11 / 11
global_ADE_all_CI_low = 0.085258
global_ADE_t50_CI_low = 0.058513
global_ADE_hard_failure_CI_low = 0.089767
global_easy_degradation_CI_high = 0.003348
global_FDE@50_CI_low = 0.148230
ETH_UCY_t50_FDE_limitation = repaired
horizon25_status = floor_non_harm_not_positive_dynamics
TrajNet_t100_status = safety_limited
metric_seconds_claim = rejected
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AH refreshes the paper claim boundary after AF/AG. The allowed claim is now stronger than Stage42-AE: global all/t50/hard remain positive, horizon=25 no longer harms, and ETH_UCY t50/FDE@50 lower bounds are positive. The remaining limitations are also explicit: horizon=25 is a floor/non-harm slice rather than positive dynamics, TrajNet|100 remains safety-limited, and metric/seconds/true-3D/foundation claims remain rejected.

Latest Stage42-AI TrajNet t100 easy-safety repair:

```text
source = fresh_run_from_stage42ag_trajnet_t100_validation_easy_safety
verdict = stage42_ai_trajnet_t100_safety_repair_pass
gates = 13 / 13
target_slice = TrajNet|100
validation_easy_nonharm_threshold = 0.0
uses_test_metrics_for_threshold = false
TrajNet100_ADE_CI_low_after = 0.048714
TrajNet100_easy_CI_high_before = 0.084984
TrajNet100_easy_CI_high_after = 0.000000
global_ADE_all_CI_low = 0.085978
global_ADE_t50_CI_low = 0.058513
global_ADE_t100_raw_frame_diagnostic_CI_low = 0.068349
global_ADE_hard_failure_CI_low = 0.090662
global_easy_degradation_CI_high = 0.001168
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AI repairs the remaining TrajNet|100 easy-safety limit with a validation-only source safety guard. This strengthens t100 raw-frame diagnostic evidence and removes the TrajNet|100 easy harm, but it still does not permit seconds-level, metric, true-3D, foundation, Stage5C, or SMC claims.

Latest Stage42-AJ post-repair paper package refresh:

```text
source = fresh_synthesis_from_stage42_ad_to_ai_artifacts
verdict = stage42_aj_post_repair_paper_package_refresh_pass
gates = 10 / 10
paper_files_refreshed = 9 / 9
included = Stage42-AD calibration, Stage42-AF horizon25 repair, Stage42-AG ETH_UCY t50/FDE repair, Stage42-AH post-repair claim matrix, Stage42-AI TrajNet t100 safety repair
metric_seconds_claim = rejected
t100_seconds_claim = rejected
Stage5C_executed = false
SMC_enabled = false
```

Stage42-AJ updates all Stage42 paper package files (`paper_outline`, `method`, `experiment_tables`, `ablation_tables`, `failure_taxonomy`, `model_card`, `data_card`, `reproducibility`, and `a_journal_gap`) with AD-AI post-repair evidence. The paper-ready scope is now stronger than Stage42-AC, but remains protected dataset-local raw-frame 2.5D.

Root-level Chinese summary requested by the user:

`/Users/yangyue/Downloads/World/README_M3W_GOAL_SUMMARY_ZH.md`

Latest Chinese detailed summary of the M3W goal:

`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/README_M3W_GOAL_DETAILED_SUMMARY_ZH.md`

Full long-form Chinese route/failure/success summary for the current goal:

`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/README_M3W_LONG_GOAL_SUMMARY_ZH.md`

It records what was tried, what failed, why it failed, what worked, and the current claim boundary, now including Stage42-A through Stage42-T. The current best deployable candidate remains M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under the Stage37/teacher floor. It is still not true 3D, not metric/seconds-level, not a foundation model, and Stage5C/SMC remain disabled. Stage42-F produced a strong protected 2.5D manuscript evidence package; Stage42-G/H added fresh ablation and sequence-history evidence; Stage42-I/J/K/L/M/N/O/P investigated full-waypoint static/context repair and explicit gain/harm selector repair. Stage42-Q identified Stage42-J/Stage42-P complementarity as report-level preflight only, Stage42-R built a local row prediction cache for validation-only combo evaluation, and Stage42-S froze that combo as a lightweight policy artifact with per-domain/per-horizon stress. Stage42-T then attempted an unseen-domain UCY transfer rule and honestly marked a blocker: the current Stage42-R row cache has no non-floor Stage42-J/P candidate predictions for UCY, so UCY remains fallback-only in this combo branch. Stage42-S passed 13/13 gates and Stage42-T passed 8/11 gates as a blocker diagnosis, but both remain dataset-local raw-frame 2.5D branch evidence and do not change the no-metric/no-Stage5C/no-SMC boundary.

Latest Stage42-R row prediction cache combo:

```text
source = fresh_run_from_row_prediction_cache
verdict = stage42_r_row_cached_combo_pass
gates = 15 / 15
cached_combo_ADE_all = 0.052387
cached_combo_ADE_t50 = 0.037934
cached_combo_ADE_t50_CI_low = 0.027740
cached_combo_ADE_hard_failure = 0.054792
cached_combo_easy_degradation = 0.001102
cached_combo_FDE_t50 = 0.100059
cache_dir = data/stage42_row_prediction_cache (not committed)
```

Latest Stage42-T UCY unseen-domain transfer diagnosis:

```text
source = fresh_run
verdict = stage42_t_ucy_transfer_blocked_no_candidate_predictions
gates = 8 / 11
validation_domains = ETH_UCY, TrajNet
unseen_test_domain = UCY
UCY_rows = 9540
UCY_all = 0.0
UCY_t50 = 0.0
UCY_hard_failure = 0.0
available_nonfloor_source_for_UCY = false
root_cause = current row cache has no non-floor Stage42-J/P candidate predictions for UCY
```

Latest Stage42-U UCY endpoint-to-full bridge audit:

```text
source = fresh_run
verdict = stage42_u_ucy_endpoint_to_full_bridge_failed_blocker
gates = 7 / 8
Stage41 pure-UCY endpoint candidate = available
row_id alignment with Stage42 full-waypoint labels = available
UCY full-waypoint ADE all = -0.070821
UCY full-waypoint ADE t50 = -0.492070
UCY full-waypoint hard/failure = -0.083302
UCY easy degradation = 0.566646
root_cause = endpoint residual success does not transfer to full-waypoint shape via linear interpolation
next = train/cache a UCY-aware full-waypoint candidate or validation-selected waypoint-shape bridge
```

Latest Stage42-V strict pure-UCY full-waypoint candidate:

```text
source = fresh_run
verdict = stage42_v_ucy_full_waypoint_candidate_pass
gates = 11 / 11
protocol = train UCY students01/students03, val UCY zara01, test UCY zara02/zara03
best_trial = ucy_full_waypoint_t50_hard
ADE_all = 0.220755
ADE_t50 = 0.290332
ADE_t50_CI_low = 0.231725
ADE_t100_raw_frame_diagnostic = 0.147461
hard_failure = 0.229484
easy_degradation = 0.000000
FDE_t50 = 0.334459
decision = deployable as a UCY full-waypoint candidate source, not yet merged into Stage42-R/S combo
```

## M3W-Neural v1 Goal Summary

Latest detailed goal-level summary:

`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/README_GOAL_SUMMARY_M3W_NEURAL_V1.md`

Current deployable candidate:

```text
model = M3W-Neural v1 composite-tail safe-switch bounded neural dynamics
safety_floor = Stage37 selector / teacher floor
gates = 41 / 41
all_ADE_improvement = 0.2103
t50_ADE_improvement = 0.1365
t100_raw_frame_diagnostic_ADE_improvement = 0.1469
hard_failure_ADE_improvement = 0.2038
easy_degradation = 0.0000
positive_external_domains = 3
Stage5C_executed = false
SMC_enabled = false
```

Main honest verdict:

M3W-Neural v1 is the strongest current protected 2.5D neural world-state candidate, but it is still not true 3D, not metric/seconds-level, and not a foundation world model. JEPA remains diagnostic-only; ungated neural dynamics remain unsafe; Stage37/teacher safety floor is still required for deployability.

## Stage42-B External Validation

Latest Stage42-B external validation:

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/external_validation_stage42.md`

Fresh external validation result:

```text
source = fresh_run
source_level_split_rebuilt = true
frozen_eval_pool_rows = 66303
frozen_eval_source_groups = 3
evaluated_rows = 55528
protected_M3W_all_ADE_improvement = 0.2103
protected_M3W_t50_ADE_improvement = 0.1365
protected_M3W_t100_raw_frame_diagnostic_ADE_improvement = 0.1469
protected_M3W_hard_failure_ADE_improvement = 0.2038
protected_M3W_easy_degradation = -0.1451
ungated_neural_all_ADE_improvement = 0.2966
ungated_neural_easy_degradation = 1.2459
stage42_b_gates = 10 / 10
verdict = stage42_b_external_validation_pass_protected_neural_not_ungated
Stage5C_executed = false
SMC_enabled = false
```

Conclusion:

Stage42-B confirms with fresh source/fold stress evidence that protected M3W-Neural v1 remains positive on external dataset-local raw-frame validation. Ungated neural still fails easy-case safety, so the Stage37/teacher floor is still required and should be treated as part of the method, not removed silently.

## Stage42-C Full-Waypoint Dynamics

Latest Stage42-C full-waypoint dynamics evidence:

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/full_waypoint_dynamics_stage42.md`

Fresh full-waypoint result:

```text
source = fresh_run
full_waypoint_sequence_model = full_trajectory_ensemble
stage42_c_gates = 12 / 12
verdict = stage42_c_full_waypoint_dynamics_pass
positive_full_waypoint_domains = ETH_UCY, TrajNet

protected_full_waypoint_ADE_all = 0.1858
protected_full_waypoint_ADE_t50 = 0.1480
protected_full_waypoint_ADE_t100_raw_frame_diagnostic = 0.2286
protected_full_waypoint_ADE_hard_failure = 0.1952
protected_full_waypoint_easy_degradation = 0.0000
protected_full_waypoint_FDE_all = 0.1938
protected_full_waypoint_FDE_t50 = 0.2158
protected_full_waypoint_near_collision_delta_005 = 0.0086

composite_tail_linear_bridge_ADE_all = 0.2103
composite_tail_linear_bridge_ADE_t50 = 0.1365
composite_tail_linear_bridge_ADE_hard_failure = 0.2038
ungated_full_waypoint_easy_degradation = 1.2459
Stage5C_executed = false
SMC_enabled = false
```

Conclusion:

Stage42-C upgrades the evidence from endpoint-only/linear bridge toward actual reconstructed future waypoint labels and all-agent world-state dynamics. The protected full-waypoint sequence model is positive on two external domains and improves t+50/t+100 raw-frame ADE/FDE while preserving easy cases and proximity gates. It does not fully replace the composite-tail linear bridge on all-ADE yet, and ungated full-waypoint neural remains unsafe.

## Stage42-A Long Research Mode Data Calibration

Latest Stage42-A data/calibration audit:

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/data_calibration_stage42.md`

Fresh audit result:

```text
source = fresh_run
datasets_audited = 7
raw_paths_found = 6
converted_paths_found = 7
external_domains_ready_from_existing_state = OpenTraj, ETH/UCY, TrajNet++, UCY
metric_claim_ready_datasets = TGSIM diagnostic only
seconds_claim_ready_datasets = none
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
stage42_b_external_validation_ready = true
stage42_c_full_waypoint_prereq_ready = true
Stage5C_executed = false
SMC_enabled = false
stage42_a_gates = 7 / 7
```

Conclusion:

Stage42 can proceed to external validation and full-waypoint dynamics from existing local converted state. It still cannot claim global metric prediction or seconds-level horizons. SDD remains pixel raw-frame; external pedestrian domains remain dataset-local raw-frame / unverified weak-metric diagnostics. TGSIM is metric traffic diagnostic only and is not pedestrian/drone world-model success.

Latest consolidated package:

`/Users/yangyue/Downloads/World/outputs/world_model_stage5_data_results`

Stage 5-Data builds on Stage 4.5. It does not start latent generative training; it creates a legal dataset registry, data-lake scaffold, causal schema, dry-run download plan, and foundation-model training gates.

Current honest verdict:

```text
model_type = pseudo-3D physics-informed learned residual state-space world model
true_3D = no
exceptional_world_model = no
expert_audit_score = 66 / 100
verdict = stage5_data_lake_partial_not_foundation_model
stage5_gates = 3 / 11
```

Key Stage 5-Data result:

```text
candidate_sources = 26
actual_converted_and_benchmarked_real_sources = 1
actual_verified_t100_real_sources = 1
strongest_causal_baseline = constant_turn_rate_velocity
best_learned_model = none for Stage5-Data dry-run
latent_stage5_ready = false
```

Main conclusion:

The data registry and Stage 5 scaffolding are now in place, but the data lake is only partial and the learned model still has not beaten the strongest causal baseline. Do not enter true Stage 5 latent generative training yet.

## Stage 5B Result

Latest Stage 5B package:

`/Users/yangyue/Downloads/World/outputs/world_model_stage5b_results`

Stage 5B converted and benchmarked actual real trajectory sources instead of counting registry-only candidates:

```text
actual_converted_real_sources = 4
actual_verified_t100_real_sources = 2
no_leakage_audit = pass
stage5b_gates = 8 / 10
expert_audit_score = 68 / 100
verdict = stage5b_usable_data_lake_but_deterministic_gate_failed
latent_stage5c_ready = false
smc_ready = false
```

Actual converted sources:

- `tgsim`: traffic, metric, verified t+100.
- `tgsim_i90`: additional public TGSIM corridor sample, traffic, metric, verified t+100.
- `trajnet`: TrajNet++ original-data Stanford subset, pedestrian-like, t+10 only in this quick conversion.
- `eth_ucy`: BIWI/ETH-style fallback from the TrajNet++ original-data tree, t+10 only in this quick conversion.

Main conclusion:

The data lake is now usable for a small multi-source benchmark, but the deterministic learned residual still does not beat strongest causal baselines on enough real sources. Do not enter Stage 5C latent generative training or enable SMC yet.

## Stage 5B.5 Result

Latest Stage 5B.5 package:

`/Users/yangyue/Downloads/World/outputs/world_model_stage5b5_results`

Stage 5B.5 built hard interaction subsets and trained a deterministic temporal-interaction fallback model. It still does not clear the deterministic gate for Stage 5C:

```text
hard_interaction_benchmark = built
torch_backend = recovered_and_checkpointed
pedestrian_drone_verified_t100_sources = 0
verified_t100_win = tgsim_i90 only
stage5b5_gates = 6 / 10
expert_audit_score = 70 / 100
verdict = stage5b5_hard_benchmark_built_but_deterministic_gate_failed
latent_stage5c_ready = false
smc_ready = false
```

Main conclusion:

The benchmark is now stricter and more honest: all-test and hard-test are separated. The PyTorch backend now runs and produced deterministic temporal-interaction checkpoints, but the model still has only one verified t+100 traffic win, no robust pedestrian/drone long-horizon source, and no hard-test gate pass. Do not enter Stage 5C latent generative training yet.

## Stage 5B.6 Result

Latest Stage 5B.6 package:

`/Users/yangyue/Downloads/World/outputs/world_model_stage5b6_results`

Stage 5B.6 repaired hard-benchmark reliability checks and trained baseline-aware gated residual deterministic models:

```text
hard_reliability_gate = failed
official_hard_gate_eligible_subsets = 0
pedestrian_drone_verified_t50_or_t100_sources = 0
gated_residual_official_target_wins = 1 dataset
alpha_gate = partially working
interaction_encoder_gate = failed
stage5b6_gates = 3 / 10
expert_audit_score = 68 / 100
verdict = stage5b6_reliability_repaired_but_deterministic_gate_failed
latent_stage5c_ready = false
smc_ready = false
```

Main conclusion:

The benchmark is now more reliable, but stricter reliability exposed that previous hard-test wins were not statistically usable. The gated residual can learn a conservative alpha, but it still does not beat strongest causal baselines on enough reliable all-test / hard-test / verified t+100 settings. Do not enter Stage 5C or enable SMC.

## Stage 6 Result

Latest Stage 6 package:

`/Users/yangyue/Downloads/World/outputs/world_model_stage6_results`

Stage 6 built HardBench-v1, BaselineFailureBench, a causal baseline-failure predictor, and a deterministic failure-aware gated residual model:

```text
pedestrian_drone_verified_t50_or_t100_sources = 0
HardBench_v1_hard_episodes = 53
HardBench_v1_gate = official
BaselineFailureBench_failure_samples = 48
failure_predictor_test_AUROC = 0.899098
failure_predictor_test_AUPRC = 0.694048
failure_aware_improvement_gate = failed
verified_long_horizon_gate = failed
interaction_gate = failed
stage6_gates = 5 / 10
expert_audit_score = 70 / 100
verdict = stage6_failure_bench_built_but_not_stage5c_ready
latent_stage5c_ready = false
smc_ready = false
```

Main conclusion:

Stage 6 proves that baseline-failure detection is learnable from causal history, but the failure-aware residual still does not improve baseline-failure cases by the required margin, verified long-horizon improvement still fails, and no pedestrian/drone long-horizon source exists locally. Do not enter Stage 5C or enable SMC.

## Stage 7 Result

Latest Stage 7 package:

`/Users/yangyue/Downloads/World/outputs/world_model_stage7_results`

Stage 7 adds scene/goal grounding without enabling latent generative modeling or SMC:

```text
scene_packs = 4
goalbench_records = 118
pedestrian_drone_verified_t50_or_t100_sources = 0
goal_predictor_test_top3 = 0.782609
majority_top3_baseline = 0.826087
best_stage7_failure_predictor_AUROC = 0.943396
stage7_gates = 5 / 10
expert_audit_score = 71 / 100
verdict = stage7_scene_goal_grounding_built_but_not_stage5c_ready
latent_stage5c_ready = false
smc_ready = false
```

Main conclusion:

Scene/goal grounding is now implemented: the system builds inferred scene packs, candidate goals, GoalBench, a goal predictor, goal/scene-conditioned failure predictors, and bounded goal-conditioned residual variants. The honest result is mixed: failure prediction improves over Stage 6, and some BaselineFailureBench/HardBench subsets improve, but GoalBench does not beat the majority top-3 baseline on test, interaction auxiliary tasks remain diagnostic, verified long-horizon improvement fails, and no pedestrian/drone t+50/t+100 source exists locally. Do not enter Stage 5C or enable SMC.

## Stage 8 Result

Latest Stage 8 package:

`/Users/yangyue/Downloads/World/outputs/world_model_stage8_results`

Stage 8 adds the Scene-Gold annotation pipeline, Scene-Gold/Silver pack builder, true multi-agent episode windows, GoalBench-Gold, Stage 8 goal/failure/world model v2, and interaction v2 ablations. It does not enable latent generative modeling or SMC:

```text
scene_gold_scenes = 0
scene_silver_scenes = 0
scene_inferred_only_scenes = 5
multi_agent_episodes_with_ge2_agents = 78
pedestrian_drone_verified_t50_or_t100_sources = 0
goal_predictor_test_top1 = 0.5
goal_predictor_majority_top1 = 0.333333
best_stage8_failure_predictor_AUROC = 0.896021
stage8_gates = 4 / 11
expert_audit_score = 71 / 100
verdict = stage8_scene_goal_multiagent_scaffold_not_stage5c_ready
latent_stage5c_ready = false
smc_ready = false
```

Main conclusion:

Stage 8 made the pipeline more world-model-shaped, but the evidence is still not strong enough. Multi-agent episodes and GoalBench-Gold now exist, but the scene annotations are still inferred-only, no real pedestrian/drone t+50/t+100 source is verified, Stage 8 failure prediction does not beat Stage 7, goal-conditioned residuals do not improve BaselineFailureBench or HardBench by the required margin, and interaction remains diagnostic. Do not enter Stage 5C or enable SMC.

## Stage 8.5 Result

Latest Stage 8.5 package:

`/Users/yangyue/Downloads/World/outputs/world_model_stage8p5_results`

Stage 8.5 is a data/annotation/per-agent preparation sprint. It does not train new residual models, does not enable latent generative modeling, and does not enable SMC:

```text
loaded_pedestrian_drone_sources = trajnet, eth_ucy
verified_pedestrian_drone_t50_or_t100_sources = 0
gold_scenes = 0
silver_scenes = 20
inferred_only_scenes = 7
per_agent_multi_agent_episodes_ge2 = 320
GoalBench_Gold_v2_official_records = 1530
stage8p5_gates = 6 / 7
expert_audit_score = 75 / 100
verdict = stage8p5_ready_for_stage9_per_agent_training
latent_stage5c_ready = false
smc_ready = false
```

Main conclusion:

Stage 8.5 successfully moves the project from primary-agent windows toward per-agent multi-agent preparation. It creates rule-confirmed silver scene annotations from train-only endpoints, builds per-agent multi-agent episodes, and creates enough official GoalBench-Gold v2 records for Stage 9. It still does not solve pedestrian/drone long-horizon t+50/t+100, and the silver labels are not human gold annotations. Stage 9 per-agent multi-agent training is now allowed; Stage 5C latent generative modeling and SMC remain disabled.

## Stage 9 Result

Latest Stage 9 package:

`/Users/yangyue/Downloads/World/outputs/world_model_stage9_results`

Stage 9 trains deterministic per-agent multi-agent scene-grounded residual models. It predicts all active agents, not only a primary agent. It still does not enable latent generative modeling or SMC:

```text
per_agent_multiagent_training = complete
predicts_all_agents = true
official_target_horizon = t+10
verified_pedestrian_drone_t50_or_t100_sources = 0
stage9_gates = 3 / 11
expert_audit_score = 75 / 100
full_model_all_test_mean_improvement = -0.001592
full_model_hard_failure_best_improvement = 0.000537
verdict = stage9_per_agent_training_done_not_stage5c_ready
latent_stage5c_ready = false
smc_ready = false
```

Main conclusion:

Stage 9 successfully trains and evaluates per-agent all-agent deterministic models, but the full scene+goal+interaction model does not beat the strongest causal baseline. Interaction and scene/goal features remain unproven for trajectory lift, easy preservation is not sufficiently assessable in the current split, and pedestrian/drone t+50/t+100 is still missing. Do not enter Stage 5C or enable SMC.

## Stage 10 Result

Latest Stage 10 package:

`/Users/yangyue/Downloads/World/outputs/world_model_stage10_results`

Stage 10 is a data acquisition, human-in-the-loop annotation, and benchmark packaging stage. It does not train a new model, does not enable latent generative modeling, and does not enable SMC:

```text
loaded_pedestrian_drone_sources = trajnet, eth_ucy
verified_pedestrian_drone_t50_or_t100_sources = 0
human_confirmed_scenes = 3
silver_rule_confirmed_scenes = 17
scene_packs_with_goals = 27
multi_agent_episodes_ge2 = 320
hard_failure_records = 309
GoalBench_v3_official_records = 1530
stage10_gates = 7 / 10
expert_audit_score = 79 / 100
verdict = stage10_ready_for_stage11_training
stage11_ready = true
latent_stage5c_ready = false
smc_ready = false
```

Main conclusion:

Stage 10 packages the current pedestrian-like data, annotation tasks, scene packs, multi-agent episodes, hard/failure records, and GoalBench v3 for the next data sprint. Three scenes have been promoted to `silver_human_confirmed`, so Stage 11 deterministic training is allowed. It still does not solve verified pedestrian/drone t+50/t+100, and these annotations are not human gold. Stage 5C latent generative modeling and SMC remain disabled.

## Stage 11 Result

Latest Stage 11 package:

`/Users/yangyue/Downloads/World/outputs/world_model_stage11_results`

Stage 11 adds AI visual-silver scene annotation from local AerialMPT image/video frames and trains a deterministic self-labeled per-agent residual model. It does not enable latent generative modeling or SMC:

```text
datasets = aerialmpt, eth_ucy, trajnet
aerialmpt_visual_silver_scenes = 14
stage11_multi_agent_episodes_ge2 = 340
verified_t10_episodes = 335
verified_t50_episodes = 0
verified_t100_episodes = 0
predicts_all_agents = true
latent_stage5c_ready = false
smc_ready = false
best_aerialmpt_improvement = -0.003994
best_eth_ucy_improvement = 0.002596
best_trajnet_improvement = 0.001107
```

Main conclusion:

Stage 11 proves the local visual annotation path works: AerialMPT frames are inspected into `ai_visual_silver` scene packs with visible image previews, observed pedestrian passage regions, and boundary-prior candidate goals. The model trains on Stage 10 self/silver labels plus AerialMPT visual-silver labels, but improvements over strongest causal baselines are tiny and AerialMPT is slightly worse than baseline. AerialMPT is pixel-space only because no homography or meter scale is present. This is not a pedestrian/drone long-horizon world model yet; verified t+50/t+100 remains missing.

## Stage 12 Result

Latest Stage 12 package:

`/Users/yangyue/Downloads/World/outputs/world_model_stage12_results`

Stage 12 adds a real pedestrian long-horizon source from the local ETH/UCY EWAP bundle, expands annotation/scene packs/GoalBench, and runs a deterministic re-benchmark. It still does not enable latent generative modeling or SMC:

```text
loaded_pedestrian_drone_sources = eth_ucy_ewap, aerialmpt, full_trajnet_original_quick
verified_pedestrian_drone_t50_or_t100_sources = eth_ucy_ewap
human_confirmed_scenes = 3
silver_rule_confirmed_scenes = 33
scene_packs_with_goals = 43
multi_agent_episodes_ge2 = 660
verified_t50_episodes = 320
verified_t100_episodes = 320
GoalBench_v4_official_records = 5574
stage12_gates = 9 / 10
expert_audit_score = 83 / 100
stage13_ready = true
latent_stage5c_ready = false
smc_ready = false
```

Deterministic re-benchmark:

```text
aerialmpt_best_improvement = -0.003994
eth_ucy_best_improvement = 0.002596
eth_ucy_ewap_t100_best_improvement = 0.0
trajnet_best_improvement = 0.001107
deterministic_5pct_gate = false
```

Main conclusion:

Stage 12 finally fixes the verified pedestrian long-horizon data blocker by adding `eth_ucy_ewap` with verified t+50/t+100. The data/annotation/GoalBench gates now allow Stage 13 deterministic training. However, the deterministic residual model still does not beat strongest causal baselines by 5%, including on EWAP t+100 where it only matches baseline. Do not enter Stage 5C or enable SMC yet.

## Final Model: BPSG-MA World Model v1

Final deliverable package:

`/Users/yangyue/Downloads/World/outputs/final_model`

The project is now packaged as a complete, runnable, evaluable final model:

```text
final_model = Baseline-Preserving Scene/Goal/Multi-Agent 2.5D World Model
short_name = BPSG-MA World Model v1
true_3D = false
large_scale_foundation_world_model = false
latent_generative = false
SMC = false
predicts_all_active_agents = true
official_horizon = t+50
t+100_status = diagnostic_small_sample
deployment_strategy = strongest_baseline_fallback_with_failure_diagnostics
expert_audit_score = 88 / 100
verdict = final_bpsg_ma_v1_delivered_with_strongest_baseline_fallback
```

Final evaluation summary:

```text
official_FDE@50_improvement_over_strongest_baseline = 0.0 for deployed fallback
Stage16 learned_correction_FDE@50_diagnostic_improvement = 0.009176
Stage16 learned_correction_FDE@100_diagnostic_improvement = 0.011476
hard_failure_improvement_gate = failed
easy_preservation = pass
scene_goal_gain = not proven
interaction_gain = not proven
```

Main conclusion:

BPSG-MA World Model v1 is a complete CPU-runnable 2.5D per-agent multi-agent world-state model with strongest-causal-baseline rollout, failure probability diagnostics, alpha/intervention decisions, bounded residual machinery, and a safety fallback. Because learned correction still does not pass the strongest causal baseline gates, the deployable final model falls back to strongest causal baselines while reporting where correction would have been attempted. This is honest and deliverable, but it is not a true 3D world model, not a foundation model, not latent generative, and not SMC.

Run:

```bash
python run_train_final_world_model.py --quick
python run_evaluate_final_world_model.py --quick
python run_select_final_model.py
python run_infer_world_model.py --demo
python run_visualize_final_world_model.py --demo
python -m pytest tests
```

## Auto-Orchestrator Status

This section is maintained by `scripts/auto_update_readme_results.py`.

```text
current_highest_stage = final_model
expert_audit_score = 88
verdict = final_bpsg_ma_v1_delivered_with_strongest_baseline_fallback
latent_generative_ready = False
smc_ready = False
learned_model_beats_strongest_baseline = 否
```

## Stage 17: Baseline Ensemble Selector

- Model status: BPSG-MA v1 remains deployable as strongest-baseline fallback with diagnostics.
- Stage17 oracle selector found per-sample baseline headroom, but trained selector/correction did not pass gates.
- Official horizon remains t+50; t+100 remains diagnostic.
- Latent generative Stage 5C and SMC remain disabled.
- Reports: `outputs/reports/report_stage17_final.md`, `outputs/reports/world_model_gate_stage17.md`.

## Stage 18: SAM-JEPA-2.5D

- Stage18 implemented self-audited multimodal JEPA representation pretraining.
- It is not true 3D, not a foundation model, not latent generative rollout, and not SMC.
- Automatic annotations are silver tiers only; gold_human remains 0.
- Quick JEPA training ran and embeddings were non-collapsed, but hard/failure correction did not pass gates.
- Official horizon remains t+50; t+100 remains diagnostic.
- Reports: `outputs/reports/report_stage18_final.md`, `outputs/reports/world_model_gate_stage18.md`.

## Stage 19: WAM-Style Data Engine

- Stage19 built a WAM-style data registry and UrbanCrowdSim2.5D curriculum data.
- Simulation is for pretraining/stress only, not real-world success.
- Egocentric/human video remains representation pretraining only and requires user-provided legal local paths.
- Official benchmark remains real top-down pedestrian/drone trajectories.
- Stage5C and SMC remain disabled.
- Reports: `outputs/reports/report_stage19_final.md`, `outputs/reports/world_model_gate_stage19.md`.
## Stage 20: Web Dataset Acquisition Agent

Stage 20 searched and registered official/candidate data sources for multimodal 2.5D world-model data acquisition. It did not train models, did not enable latent generative Stage 5C, and did not enable SMC.

```text
candidate_sources = 33
successful_auto_download_sources = 0
successful_local_path_verifications = 6
successful_converted_sources = 5
new_official_topdown_benchmark_sources = 0
stage20_gates = 9 / 11
latent_stage5c_ready = false
smc_ready = false
verdict = stage20_web_dataset_acquisition_package_built_stage5c_blocked
```

Main conclusion:

Stage 20 built the web-search registry, license audit, dry-run download plan, local-path verification, and data-acquisition package. The project still needs user-provided SDD/OpenTraj/full ETH-UCY paths for a stronger real top-down pedestrian/drone benchmark.

## Stage 21: User-Provided OpenTraj and SDD Intake

The user provided OpenTraj and Kaggle Stanford Drone Dataset sources. OpenTraj was fetched from the GitHub source into `external_data/OpenTraj`, and the user-provided `external_data/archive.zip` was extracted as a local Stanford Drone Dataset mirror. Raw external data remains ignored by git.

```text
opentraj_local = true
sdd_archive_found = true
sdd_scenes = 8
sdd_videos = 60
sdd_tracks = 10300
sdd_annotation_rows = 10616256
sdd_raw_frame_t50_samples = 10101593
sdd_raw_frame_t100_samples = 9589470
coordinate_status = pixel-space
metric_status = no homography/scale verified
latent_stage5c_ready = false
smc_ready = false
```

Main conclusion:

The next concrete step is full SDD world-state conversion with scene split, train-only goal dictionaries, causal velocity, horizon audit, and no-leakage validation. The raw-frame t+50/t+100 counts are real annotation-frame availability, but they are not metric-world claims and not yet full official benchmark results.

## Stage 21: SDD World-State Conversion

The user-provided Stanford Drone Dataset archive has been converted into per-video pixel-space world-state shards. This is not a metric world model claim: no homography/scale has been verified yet, and effective seconds for t+50/t+100 are not claimed until FPS audit.

```text
sdd_world_state_rows = 10616256
sdd_tracks = 10300
sdd_scenes = 8
sdd_videos = 60
sdd_raw_frame_t50_samples = 10009005
sdd_raw_frame_t100_samples = 9497463
scene_level_split = train 40 videos / val 4 videos / test 16 videos
no_leakage_audit = pass
coordinate_status = pixel-space
metric_status = no homography/scale verified
latent_stage5c_ready = false
smc_ready = false
```

Main conclusion:

Stage 21 turns the user-provided SDD archive into usable local world-state shards and confirms raw-frame long-horizon availability. The next step is to build per-agent multi-agent episodes, train-only candidate goals, scene packs, and then re-benchmark deterministic baselines/head models without claiming metric performance.

## Stage 22: SDD Official Pixel-Space Benchmark

Stage 22 builds a quick SDD pixel-space benchmark from the user-provided Stanford Drone Dataset archive: scene packs, lazy per-agent episodes, no-leakage audit, causal baselines, HardBench/BaselineFailureBench, GoalBench, existing-model transfer eval, and quick SDD selector/failure/JEPA probes. It does not enable latent generative Stage 5C or SMC.

```text
SDD_official_pixel_space_benchmark = True
SDD_scene_packs = built
SDD_episode_windows_quick = 27600
SDD_t50 = official pixel raw-frame
SDD_t100 = official pixel raw-frame / diagnostic seconds unknown
selector_effective = False
failure_predictor_effective = False
JEPA_effective = False
correction_effective = False
latent_stage5c_ready = false
smc_ready = false
verdict = stage22_sdd_pixel_benchmark_built_training_heads_not_stage5c_ready
```

Main conclusion:

SDD is now usable as a real top-down pixel-space benchmark, but quick SDD-specific learned heads did not pass selector/correction/JEPA gates. Do not claim metric performance, true 3D, foundation-model status, Stage 5C readiness, or SMC readiness.

## Stage 23: SDD Dual-Split Quick-Plus Benchmark

Stage 23 adds SDD dual-split evaluation: cross-scene generalization and within-scene video split for scene/goal learning. The run completed in `quick-plus` mode, not full medium, and must not be reported as medium/full.

```text
current_model_type = 2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold
true_3D = false
foundation_world_model = false
SDD_coordinate_status = pixel-space
SDD_horizon_status = raw annotation-frame; effective seconds unknown
dual_split_built = True
medium_benchmark_built = False
quick_plus_benchmark_built = True
selector_effective = False
failure_predictor_effective = False
JEPA_effective = False
correction_effective = False
latent_stage5c_ready = false
smc_ready = false
verdict = stage23_sdd_quick_plus_dual_split_benchmark_heads_not_stage5c_ready
```

Main conclusion: dual-split SDD evaluation infrastructure is now in place, but validation-selected selector, failure predictor, JEPA, and correction specialist still do not clear the deterministic gates in quick-plus mode.

## Stage 24: SDD Fast Cache and Medium Selector Training

Stage 24 fixes the SDD compressed-NPZ random I/O bottleneck with a per-video uncompressed `.npy` memmap cache and track/frame indexes, then runs the medium/medium-lite benchmark path without falling back to quick-plus.

```text
true_3D = false
foundation_world_model = false
SDD_coordinate_status = pixel-space
SDD_horizon_status = raw annotation-frame; effective seconds unknown
fast_cache_built = true
true_medium_status = 是
selector_effective = False
failure_predictor_effective = True
JEPA_effective = False
correction_effective = False
latent_stage5c_ready = false
smc_ready = false
verdict = stage24_sdd_fast_cache_medium_run_heads_not_stage5c_ready
```

## Stage 25: Selector Failure Forensics and Regret-Minimizing Baseline Policy

Stage 25 diagnoses why the Stage 24 hard-label selector failed despite large oracle headroom, then replaces hard classification with regret-aware expected-FDE selection plus confidence/gain/easy fallback gates.

```text
true_3D = false
foundation_world_model = false
SDD_coordinate_status = pixel-space
SDD_horizon_status = raw annotation-frame; effective seconds unknown
selector_root_cause_identified = True
regret_selector_effective = False
soft_label_selector_effective = False
hierarchical_selector_effective = False
failure_assisted_selector_effective = False
easy_preserved = True
final_model_v1_2_upgraded = False
latent_stage5c_ready = false
smc_ready = false
verdict = stage25_selector_forensics_regret_policy_executed_not_stage5c_ready
```

## Stage 26: Feature-Complete Cost-Aware SDD Baseline Selector Training

Stage 26 builds a causal SDD feature store from the Stage24 medium baseline-evaluated windows and trains expected-FDE/risk selectors with conservative fallback. It does not train JEPA, residual correction, latent generative rollout, or SMC.

```text
true_3D = false
foundation_world_model = false
SDD_coordinate_status = pixel-space
SDD_horizon_status = raw annotation-frame; effective seconds unknown
feature_store_built = True
selected_model = stage26_failure_assisted_selector
t50_improvement = 0.14583655843823773
hard_failure_improvement = 0.11234058960663984
easy_degradation = 0.01808836280803794
latent_stage5c_ready = false
smc_ready = false
verdict = stage26_feature_complete_cost_aware_selector_executed_not_stage5c_ready
```

## M3W: Real-World Multimodal Agent-Scene World Model

M3W local-small adds JEPA-only, Transformer-only, and JEPA+Transformer hybrid code, then executes on the Stage26 SDD causal feature store. The PyTorch backend executed with the arm64 `.venv-pytorch` runtime requirement; this is still local-small / evidence sprint, not medium/full. It does not execute latent generative Stage5C or SMC.

```text
true_3D = false
foundation_world_model = false
SDD_coordinate_status = pixel-space
SDD_horizon_status = raw annotation-frame; effective seconds unknown
M3W_execution_backend = torch_arm64_cpu_multithread
M3W_variant = hybrid
M3W_t50_improvement = 0.1308150291442871
M3W_hard_failure_improvement = 0.10240167379379272
M3W_easy_degradation = 0.010665178298950195
beats_stage26_selector = False
latent_stage5c_ready = false
smc_ready = false
verdict = m3w_stage27_evidence_executed_not_ccfa_candidate_stage26_remains_best
```

## Stage 28: M3W-LAS Evidence Sprint

Stage 28 tests whether frozen M3W JEPA/Transformer/Hybrid latents improve the Stage26 cost-aware selector. It does not execute Stage5C, SMC, ordinary residual correction, metric conversion, or seconds-level horizon claims.

```text
true_3D = false
foundation_world_model = false
SDD_coordinate_status = pixel-space
SDD_horizon_status = raw annotation-frame; effective seconds unknown
best_variant = all_latent
t50_improvement = 0.1686288243790961
hard_failure_improvement = 0.1336398986813968
easy_degradation = 0.01928694490688554
candidate_v2 = True
stage5c_ready = false
smc_ready = false
verdict = stage28_m3w_las_candidate_v2_not_stage5c_ready
```

## M3W Long-Term State Machine

The M3W state machine freezes the Stage28 M3W-LAS v2 candidate and advances through gated stages without enabling Stage5C or SMC.

```text
current_state_machine_stage = F_plan_generated
true_3D = false
foundation_world_model = false
SDD_coordinate_status = pixel-space
SDD_horizon_status = raw annotation-frame; effective seconds unknown
stage5c_executed = false
smc_enabled = false
```

## Stage30: Fresh Recompute, External Validation, Raw Audit

Stage30 reruns M3W-LAS v2 verification with explicit source labels: `fresh_run`, `cached_verified`, and `not_run`. It refits ablations for seeds 0/1/2, runs 3000 bootstrap, attempts non-SDD OpenTraj conversion, audits raw SDD timing/geometry, and keeps Stage5C/SMC disabled.

```text
true_3D = false
foundation_world_model = false
SDD_coordinate_status = pixel-space
stage5c_executed = false
smc_enabled = false
gates = 14 / 14
verdict = stage30_fresh_recompute_verified_m3w_las_v2_candidate_not_stage5c_ready
```

## Stage31: External Topdown Generalization

Stage31 converts non-SDD OpenTraj pedestrian subsets into the M3W-LAS feature-store schema, builds an external latent cache from frozen M3W checkpoints, evaluates zero-shot transfer, runs bounded external selector-head adaptation, and reports domain gap without enabling Stage5C or SMC.

```text
true_3D = false
foundation_world_model = false
external_coordinates = dataset-local / unverified weak metric diagnostic
stage5c_executed = false
smc_enabled = false
gates = 10 / 11
verdict = stage31_external_domain_gap_sdd_candidate_only
```

## Stage32: External Domain Alignment

Stage32 audits the Stage31 external domain gap, builds multiple domain normalizations, recomputes normalized external baselines, measures latent distribution shift, trains domain-adapted selectors, and evaluates SDD/external cross-domain transfer without enabling Stage5C or SMC.

```text
true_3D = false
foundation_world_model = false
external_coordinates = dataset-local / unverified weak metric diagnostic
stage5c_executed = false
smc_enabled = false
gates = 9 / 11
verdict = stage32_domain_alignment_partial_not_cross_domain_candidate
```

Key Stage32 outcome:

- External conversion, no-leakage, normalization, baseline reaudit, latent alignment, selectors, cross-domain matrix, and gates were run as `fresh_run`.
- Best external adapted selector fell back to the external strongest causal baseline: all improvement `0.0`, t+50 improvement `0.0`.
- SDD-trained zero-shot selector still failed externally: all improvement `-0.337476`, t+50 improvement `-1.018801`.
- Mixed-domain selector preserved positive SDD average improvement but failed easy preservation, so it is not deployable.
- Cross-dataset world-model generalization is not established; M3W-LAS remains an SDD pixel raw-frame candidate.
- Tests: `python -m pytest tests` -> `58 passed`.

## Stage33: Coordinate-Invariant Cross-Domain M3W

Stage33 rebuilds the external transfer stack around train-only external scene/goal context, coordinate-invariant tokens, relative-error baseline targets, latent domain adapters, and domain-conditioned selectors. It does not execute Stage5C or enable SMC.

```text
true_3D = false
foundation_world_model = false
sdd_coordinates = pixel raw-frame
external_coordinates = dataset-local / unverified weak metric diagnostic
stage5c_executed = false
smc_enabled = false
best_external_all_improvement = 0.0
best_external_t50_improvement = 0.0
gates = 11 / 13
verdict = stage33_coordinate_invariant_partial_not_cross_domain_candidate
```

Key Stage33 outcome:

- External scene packs and train-only goal context were built as geometry proxies; no test endpoints were used.
- Coordinate-invariant features, relative-FDE targets, relative baselines, latent domain adapter, domain-conditioned selectors, and cross-domain matrix were run as `fresh_run`.
- Domain adapter reduced latent mean-distance, but selector deployment still fell back to strongest baselines externally.
- Best external all/t+50 improvement remained `0.0`; this is safe fallback, not positive cross-domain transfer.
- External-only relative selectors still damaged easy cases, so they are not deployable.
- Cross-domain world-model candidate gate failed; M3W-LAS remains SDD pixel raw-frame candidate plus external diagnostic evidence.
- Tests: `python -m pytest tests` -> `61 passed`.

## Stage34: External Row Geometry and Domain-Conditioned Transfer

Stage34 reconstructs external per-row geometry from raw OpenTraj/TrajNet rows, builds train-only goals and per-row goal distance/angle features, recomputes relative baselines v2, and evaluates domain-conditioned transfer. Stage5C and SMC remain disabled.

```text
true_3D = false
foundation_world_model = false
sdd_coordinates = pixel raw-frame
external_coordinates = dataset-local / unverified weak metric diagnostic
stage5c_executed = false
smc_enabled = false
best_external_all_improvement = 0.0
best_external_t50_improvement = 0.0
gates = 9 / 13
verdict = stage34_row_geometry_done_no_external_positive_transfer
```

Key Stage34 outcome:

- External row geometry was reconstructed and aligned for all Stage31 external rows: train `119109`, val `7685`, test `3636`.
- Future endpoint coordinates are stored only as supervision/evaluation labels and are not used as inference features.
- Train-only goals and per-row goal distance/angle features were built without test endpoints.
- Diagnostic external selector lift exists on t+50 (`~6.6%`) and hard/failure (`~25.1%`), but it is not deployable because all-test is negative and easy degradation is high.
- Latent adapter reduced latent distance but did not produce predictive lift.
- Cross-domain candidate gate remains failed; this is not external positive transfer.
- Tests: `python -m pytest tests` -> `64 passed`.

## Stage35: External Selective Transfer

Stage35 expands local non-SDD top-down pedestrian data where safely parseable, builds scene-level external splits, hard/easy/failure labels, selective transfer policies, and external selector v3. It does not execute Stage5C or enable SMC.

```text
true_3D = false
foundation_world_model = false
sdd_coordinates = pixel raw-frame
external_coordinates = dataset-local / unverified weak metric diagnostic
stage5c_executed = false
smc_enabled = false
best_external_all_improvement = 0.12131890857784355
best_external_t50_improvement = 0.0
best_external_hard_improvement = 0.1398494448930071
best_external_easy_degradation = 0.0004114683717719725
gates = 12 / 14
verdict = stage35_external_selective_transfer_not_deployable
```

Key Stage35 outcome:

- External data expansion converted `18` local non-SDD track files into split v2 rows: train `158942`, val `112746`, test `66303`.
- Test horizons include t+50 `16263` and t+100 `10008` dataset-local raw-frame rows; these are not metric/seconds claims.
- External hard/easy/failure labels were built with oracle headroom around `52.9%` on test.
- Selective transfer improved all-test by `0.12131890857784355` and hard/failure by `0.1398494448930071` while easy degradation stayed `0.0004114683717719725`.
- t+50 improvement stayed `0.0`, so Stage35 is not a deployable cross-domain M3W candidate.
- Tests: `python -m pytest tests` -> `67 passed`.

## Stage36: External t+50 Transfer Repair

Stage36 focuses only on the Stage35 blocker: external t+50 transfer. It builds t+50 forensics, horizon-specific features, horizon selectors, t+50 conservative policy search, bounded t+50 curriculum, cross-domain eval v4, and failure analysis. It does not execute Stage5C or enable SMC.

```text
true_3D = false
foundation_world_model = false
external_coordinates = dataset-local / unverified weak metric diagnostic
stage5c_executed = false
smc_enabled = false
final_all_improvement = 0.12131890857784355
final_t50_improvement = 0.0
final_t100_diagnostic_improvement = 0.0
final_hard_improvement = 0.1398494448930071
final_easy_degradation = 0.0004114683717719725
gates = 12 / 14
verdict = stage36_t50_transfer_not_repaired
```

Key Stage36 outcome:

- t+50 forensics confirmed `16263` external t+50 test rows and real oracle headroom, but Stage35 t+50 switch rate was `0.0`.
- Horizon-specific t+50 selectors and bounded curriculum were trained/validated, but no policy safely passed the `>3%` t+50 gate on held-out test scenes.
- all/hard/easy remain acceptable through conservative fallback, but t+50 remains unrepaired, so Stage36 is not deployable cross-domain M3W.
- Tests: `python -m pytest tests` -> `70 passed`.

## Stage37: External t+50 Causal History Transfer

Stage37 builds past-only external history windows and scene-agnostic goal prototypes to repair the external t+50 gate. It does not execute Stage5C or enable SMC.

```text
true_3D = false
foundation_world_model = false
external_coordinates = dataset-local / unverified weak metric diagnostic
stage5c_executed = false
smc_enabled = false
final_all_improvement = 0.1348254070727205
final_t50_improvement = 0.08457292542209705
final_t100_diagnostic_improvement = 0.0
final_hard_improvement = 0.1554340386904196
final_easy_degradation = 0.0004114683717719725
gates = 16 / 16
verdict = stage37_t50_transfer_repaired_deployable
```

Key Stage37 outcome:

- Built K=8/16/32/64 past-only history windows and scene-agnostic goal prototypes from train/past motion only.
- Rebuilt t+50 candidate baseline family and switchability models.
- t+50 now passes the Stage37 external gate under dataset-local raw-frame evaluation, but no metric/seconds/3D claim is made.
- Tests: `python -m pytest tests` -> `73 passed`.

## Stage38: External Robustness and Safe Dynamics Head

Stage38 freezes the Stage37 deployable selector, audits external domain coverage, evaluates frozen external generalization, trains bounded correction/dynamics heads under Stage37 fallback, and reports statistical evidence. It does not execute Stage5C or enable SMC.

```text
true_3D = false
foundation_world_model = false
external_coordinates = dataset-local / unverified weak metric diagnostic
stage5c_executed = false
smc_enabled = false
stage37_all_improvement = 0.1348254070727205
stage37_t50_improvement = 0.08457292542209705
stage37_hard_improvement = 0.1554340386904196
stage37_easy_degradation = 0.0004114683717719725
correction_deployment = keep_stage37_selector
gates = 14 / 15
verdict = stage38_robustness_partial_keep_stage37_selector
```

Key Stage38 outcome:

- Stage37 policy is frozen and remains the current external best unless bounded correction beats it safely.
- UCY held-out remains positive; ETH/TrajNet held-out external tests are honest blockers under the frozen split.
- Bounded correction/dynamics head is trained and evaluated with fallback; failed correction is not deployed.
- Tests: `python -m pytest tests` -> `77 passed in 8.02s`.

## Stage39: Stage37-Protected Neural World Dynamics

Stage39 trains real causal Transformer, JEPA auxiliary, and Hybrid neural dynamics heads under the frozen Stage37 safety floor. Neural outputs are diagnostic unless they beat Stage37 under fallback while preserving easy cases. Stage5C and SMC remain disabled.

```text
true_3D = false
foundation_world_model = false
external_coordinates = dataset-local / unverified weak metric diagnostic
raw_frame_horizons = true
stage5c_executed = false
smc_enabled = false
deployment_decision = keep_stage37_selector
best_neural = Transformer_only
neural_beats_stage37 = False
gates = 11 / 13
verdict = stage39_neural_dynamics_diagnostic_keep_stage37
```

Key Stage39 outcome:

- Stage37 safety floor is frozen and remains the external deployment floor.
- Transformer/JEPA/Hybrid neural dynamics are trained with arm64 `.venv-pytorch` runtime, single-process data loading, checkpoints, and heartbeat files.
- ETH/TrajNet held-out repair remains an honest blocker unless a new split protocol is built.
- Tests: `python -m pytest tests` -> `80 passed in 9.11s`.

## Stage40: Neural World Dynamics Optimization

Stage40 diagnoses Stage39 neural failure, rebuilds the training target around Stage37 safety mechanisms, trains candidate-ranker neural dynamics trials with teacher/safety distillation, runs bounded optimization, and evaluates against the frozen Stage37 selector. Stage5C and SMC remain disabled.

```text
true_3D = false
foundation_world_model = false
stage5c_executed = false
smc_enabled = false
trained_neural_world_model = true
neural_exceeds_stage37 = False
deployment_decision = keep_stage37_selector
best_stage40_neural = Stage40_causal_transformer_candidate_ranker
gates = 11 / 12
verdict = stage40_neural_optimization_keep_stage37
```

Key Stage40 outcome:

- Neural models were trained and optimized, not merely planned.
- Deployment remains Stage37 selector unless Stage40 neural beats the same-subset Stage37 floor.
- Tests: `python -m pytest tests` -> `83 passed in 9.30s`.

## Stage41: M3W Neural World Model Breakthrough Attempt

Stage41 rebuilds the external split so test is no longer UCY-only, constructs a seq2seq neural world-model dataset from past-only history windows, trains Transformer / JEPA-only / Hybrid / mixture-style neural dynamics trials, runs validation-selected safety policies, and compares against the Stage37 deployable floor. Stage5C and SMC remain disabled.

```text
true_3D = false
foundation_world_model = false
stage5c_executed = false
smc_enabled = false
trained_neural_world_model = true
deployment_decision = stage41_protected_neural_candidate_pending_user_acceptance
neural_exceeds_stage37_by_gate_margin = True
positive_external_domains = 3
best_stage41_neural = fresh_self_gated_endpoint::binary_fde_neural_dynamics
gates = 41 / 41
verdict = stage41_self_gated_neural_candidate_endpoint_geometry_verified
```

Key Stage41 caveat: the rebuilt external dataset initially used row-level per-agent history plus neighbor aggregates. A second pass added all-agent same-frame neighbor tokens and endpoint-risk neural trials. The fresh self-gated endpoint candidate now beats the Stage37/source-rotation floor in all/t50/hard with easy preserved. The Stage41 endpoint geometry audit now verifies that safety-floor endpoint deltas and FDE labels are aligned, so continuous endpoint interpolation is geometry-aligned but remains protected by the safety floor.

Stage41 second pass:

- all-agent dataset: train 80k / val 24k / test 34,777 rows with up to 6 same-frame agents and past-only history tokens.
- best all-agent neural: `all_agent_t100_curriculum`.
- result: all improvement `-5.81288021355153e-05`, t+50 `0.0`, hard/failure `-3.246737611095618e-05`, easy degradation `0.0009077552127720878`.
- deployment remains `keep_stage37_selector`.
- intervention calibrator: `calibrated_all_agent_endpoint_t100_focus` with deployment `keep_stage37_selector`.
- t50 rescue: `calibrated_all_agent_endpoint_easy_guard::t50_short_history_guard` with deployment `keep_stage37_selector`.
- policy blender: `metadata_guarded` with deployment `keep_stage37_selector`.
- candidate-FDE distiller: `candidate_distill_t100_curriculum::balanced` with deployment `keep_stage37_selector`.
- validation gap audit: blockers `['ETH_UCY t50 validation headroom is not representative: val=0.0445, test=0.4881']`; stratified candidate status `candidate_protocol_not_used_for_stage41_claims`.
- stratified protocol candidate: `stratified_long_horizon::balanced` with deployment `keep_stage37_selector` and t50 `0.09084803055127988`.
- locked-v2 confirmatory: deployment `keep_stage37_selector`, stable margin `False`, t50 mean `0.11543840676955701`.
- locked-v2 tail-robust: deployment `keep_stage37_selector`, stable margin `False`, t50 mean `0.11007864650973702`.
- locked-v2 hard/all: deployment `keep_stage37_selector`, stable margin `False`, hard mean `0.10834594718434147`.
- locked-v2 domain-focused experts: deployment `keep_stage37_selector`, summary `{'ETH_UCY': {'best_trial': 'locked_v2_eth_hard_expert_seed0', 'best_mode': 'domain_hard', 'domain_test_metrics': {'rows': 26092, 'all_improvement': 0.01129205281700052, 't50_improvement': 0.0004292731375977743, 't100_improvement': 0.002026249305847494, 'hard_failure_improvement': 0.010058122841057004, 'easy_degradation': 0.0, 'switch_rate': 0.018473095201594358}}, 'UCY': {'best_trial': 'locked_v2_ucy_hard_expert_seed0', 'best_mode': 'domain_hard', 'domain_test_metrics': {'rows': 13254, 'all_improvement': 0.05172065705329609, 't50_improvement': 0.12630349635484828, 't100_improvement': 0.003960889298421644, 'hard_failure_improvement': 0.05540953335396692, 'easy_degradation': 0.0, 'switch_rate': 0.03945978572506413}}, 'TrajNet': {'best_trial': 'locked_v2_trajnet_hard_expert_seed101', 'best_mode': 'domain_tail', 'domain_test_metrics': {'rows': 17193, 'all_improvement': 0.18067366523521744, 't50_improvement': 0.210138445350176, 't100_improvement': 0.26814367001244144, 'hard_failure_improvement': 0.20566244426479086, 'easy_degradation': 0.0024923440564112997, 'switch_rate': 0.13261210957948003}}}`.
- locked-v2 domain expert composer: deployment `keep_stage37_selector`, margin result `False`, hard `0.11515500294044723`.
- locked-v2 neural ensemble: deployment `keep_stage37_selector`, margin result `False`, t50 `0.17139836347088577`.
- locked-v2 relaxed easy-budget: deployment `candidate_needs_fresh_confirmation_before_deployment`, margin result `True`, all `0.2020717215500809`, t50 `0.2570819493873687`, hard `0.21034517907168104`. This is candidate evidence requiring fresh confirmation before deployment.
- locked-v2 domain-safe relaxed: deployment `candidate_needs_fresh_confirmation_before_deployment`, margin result `True`, all `0.1707426681634402`, t50 `0.23639645488658112`, hard `0.17761634616412003`, max domain easy `0.0055509018258728116`. This fixes the ETH_UCY easy-risk issue but still requires fresh confirmation before deployment.
- locked-v2 fixed-policy confirmation: deployment `candidate_needs_fresh_external_confirmation_before_deployment`, margin `True`, stress `True`, fresh confirmation `False`, all `0.1707426681634402`, t50 `0.23639645488658112`, hard `0.17761634616412003`.
- source-rotation fresh confirmation: deployment `stage41_neural_fresh_confirmed_partial_not_full_replacement`, fresh pass `True`, full replacement `False`, all `0.20881762937561832`, t50 `0.05448600669657733`, t100 `0.4572355026149352`, hard `0.22538184888542845`, easy `0.0`, t50 oracle ceiling `0.07570014620278032`. This confirms all/hard neural lift on fresh held-out source files but does not fully replace Stage37 because t50 remains below Stage37.
- fresh residual endpoint candidate: deployment `candidate_residual_full_replacement_pending_user_acceptance`, full replacement `True`, vs-floor all `0.2939852883056493`, t50 `0.3446657894513262`, t100 `0.45728888926366984`, hard `0.31812644745232777`, easy `0.0`, vs-source-rotation-base t50 `0.30691223004008084`, unprotected endpoint easy `1.3360420821946413`. This is the first Stage41 neural residual candidate that clears all/t50/hard on fresh rotation, but it must remain protected because unprotected endpoint still hurts easy cases.
- fresh bounded residual candidate: deployment `diagnostic_keep_stage37_floor`, protected full replacement `False`, no-fallback safe `False`, vs-floor all `0.2093691451653562`, t50 `0.05492974017452401`, hard `0.22598240425938143`, unprotected easy `20.566047225938334`. This clipped residual hypothesis did not fix no-fallback safety and remains diagnostic.
- fresh endpoint interpolation candidate: deployment `diagnostic_keep_stage37_floor`, protected full replacement `True`, no-fallback safe `False`, alpha `1.0`, vs-floor all `0.4360308896303905`, t50 `0.462871212116444`, t100 `0.37849164106051136`, hard `0.4461783076319382`, easy `0.0`, vs-source-rotation-base all `0.2871819048185579`, t50 `0.43191873236381895`, unprotected easy `0.24986518840352323`. This is the strongest protected neural evidence so far, but no-fallback safety remains false.
- fresh endpoint gain-gate candidate: deployment `diagnostic_keep_stage37_floor`, protected full replacement `False`, positive neural switch `True`, vs-floor all `0.4434591531399067`, t50 `0.46944827207913975`, t100 `0.46564609285948366`, hard `0.45984844420064597`, easy `0.0`, vs-source-rotation-base all `0.29655953283228076`, t50 `0.4388833849446364`, t100 `0.015398991158436237`, switch `0.4499891946405417`, ungated easy `1.3360420821946413`. This is the strongest protected neural dynamics evidence so far and directly fixes the ungated endpoint easy/t100 failure through a learned gain/harm gate.
- fresh self-gated endpoint candidate: deployment `self_gated_m3w_neural_v1_candidate_pending_user_acceptance`, protected full replacement `True`, no-external-fallback safe `True`, vs-floor all `0.41964214194307703`, t50 `0.4061979981406123`, t100 `0.45728888926366984`, hard `0.43608295876101655`, easy `0.0`, self-gated vs source-rotation-base all `0.26645599312381374`, t50 `0.37198928631867734`, t100 `0.0`, hard `0.2719897698942525`, easy `0.0`, raw ungated t100 `-0.007687587556248099`, raw ungated easy `1.3360420821946413`. This fixes the Gate10 no-external-fallback safety check through an internal binary neural gate, while still recording that continuous endpoint interpolation is pending floor-geometry repair.
- Tests: `python -m pytest tests` -> `107 passed in 66.97s`.

<!-- M3W_NEURAL_V1:START -->
## M3W-Neural v1 Frozen Evidence Package

Stage41 evidence is now frozen into `outputs/m3w_neural_v1/` as a cached-verified M3W-Neural v1 candidate package.

```text
true_3D = false
foundation_world_model = false
metric_claim = false
seconds_level_claim = false
stage5c_executed = false
smc_enabled = false
gates = 41 / 41
all_improvement = 0.2102513255185352
t50_improvement = 0.13652231450154184
t100_raw_frame_diagnostic = 0.14694086716388166
hard_failure_improvement = 0.20384916307933942
easy_degradation = 0.0
positive_external_domains = 3
pure_ucy_source_heldout_gate = True
pure_ucy_three_way_train_val_test_gate = False
strict_pure_ucy_neural_retrain_gate = True
strict_pure_ucy_neural_best_trial = pure_ucy_transformer
strict_pure_ucy_neural_best_mode = bounded_endpoint_residual
strict_pure_ucy_neural_all_improvement = 0.09008338741058997
strict_pure_ucy_neural_t50_improvement = 0.08800918427966675
strict_pure_ucy_neural_hard_failure_improvement = 0.09358425348430643
strict_pure_ucy_neural_easy_degradation = 0.0
strict_pure_ucy_neural_remaining_blocker = 
strict_pure_ucy_neural_statistically_stable = True
strict_pure_ucy_neural_bootstrap_lows_all_t50_t100_hard = 0.08888491295965614 / 0.08630755189944371 / 0.08069804446646152 / 0.09229635491900144
endpoint_to_full_bridge_gate = True
endpoint_to_full_bridge_positive_domains = ['ETH_UCY', 'TrajNet']
endpoint_to_full_statistical_gate = True
endpoint_to_full_statistical_positive_domains = ['ETH_UCY', 'TrajNet']
required_ablation_coverage_gate = True
required_ablation_cross_protocol_limitations = []
same_protocol_architecture_ablation_gate = True
same_protocol_best_protected_architecture = Stage41_fresh_self_gated_endpoint_candidate
same_protocol_transformer_only_deployable = False
same_protocol_jepa_only_deployable = False
same_protocol_hybrid_deployable = False
calibrated_learned_shape_meta_gate = True
calibrated_learned_shape_positive_domains = ['ETH_UCY', 'TrajNet']
composite_tail_evidence_pass = True
composite_tail_multiseed_pass = True
all_agent_composite_world_state_pass = True
all_agent_composite_ade_all_improvement = 0.2102513255185352
all_agent_composite_ade_t50_improvement = 0.13652231450154184
all_agent_composite_fde_all_improvement = 0.19816955620782206
all_agent_composite_fde_t50_improvement = 0.17387876491801468
fixed_prior_source_switch_beats_fixed = False
fixed_prior_residual_oracle_headroom = False
deployment_state = composite_tail_candidate_pending_final_package_acceptance
```

Current best candidate: M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under the Stage37/teacher safety floor. Stage37 remains the explicit fallback floor, and ungated/full-row neural dynamics are not claimed safe.
<!-- M3W_NEURAL_V1:END -->

<!-- M3W_NEURAL_COMPLETION_AUDIT:START -->
## M3W-Neural v1 Completion Audit

The Stage41 breakthrough objective now has complete protected M3W-Neural v1 evidence under its stated gates. M3W-Neural v1 has a no-base-switch joint policy distiller with bootstrap/multi-seed stability, train-only UCY repair, grouped all-agent rollout consistency, neural group-consistency distillation, teacher-guided proposal repair, domain-local endpoint retrain, endpoint-to-full statistical bridge evidence, same-protocol architecture ablation evidence, and a goal-level completion audit. The rollout is still not a latent generative world state.

```text
completion_status = complete
all_agent_repair_all = 0.09976285280545372
all_agent_repair_t50 = -0.002800354643290648
all_agent_repair_t100_diagnostic = 0.26476770940707695
all_agent_repair_hard_failure = 0.10663942185551323
all_agent_repair_easy = 0.0
all_agent_deployment = diagnostic_keep_m3w_neural_v1_endpoint_candidate
all_agent_t50_specialist_t50 = 0.09375204966816386
all_agent_t50_specialist_all = 0.023127391643180673
all_agent_t50_specialist_hard = 0.02472403070303797
all_agent_t50_specialist_easy = 0.0
all_agent_t50_specialist_deployment = diagnostic_keep_m3w_neural_v1_endpoint_candidate
all_agent_policy_composer_variant = risk_all_t50_override
all_agent_policy_composer_all = 0.12271025390115187
all_agent_policy_composer_t50 = 0.0902220631231122
all_agent_policy_composer_t100_diagnostic = 0.26476770940707695
all_agent_policy_composer_hard = 0.13117103605560632
all_agent_policy_composer_easy = 0.0
all_agent_policy_composer_deployment = diagnostic_keep_m3w_neural_v1_endpoint_candidate
all_agent_locked_v2_all = 0.17133358603743754
all_agent_locked_v2_t50 = 0.2368852639341944
all_agent_locked_v2_t100_diagnostic = 0.19179621162648797
all_agent_locked_v2_hard = 0.1782452385355816
all_agent_locked_v2_easy = 0.0
all_agent_locked_v2_stage37_margin_pass = True
all_agent_locked_v2_stress_pass = True
all_agent_locked_v2_fresh_confirmation_pass = False
fresh_all_agent_endpoint_best = fresh_all_agent_endpoint_ensemble
fresh_all_agent_endpoint_all = 0.26231894385271437
fresh_all_agent_endpoint_t50 = 0.2754049021317291
fresh_all_agent_endpoint_t100_diagnostic = 0.3011547102800356
fresh_all_agent_endpoint_hard = 0.2850203073377977
fresh_all_agent_endpoint_easy = 0.0
fresh_all_agent_endpoint_positive_domains = 2
fresh_all_agent_endpoint_deployment = fresh_all_agent_endpoint_candidate_needs_independent_acceptance
full_trajectory_world_state_best = full_trajectory_ensemble
full_trajectory_world_state_all = 0.18577852429834418
full_trajectory_world_state_t50 = 0.14803699577731477
full_trajectory_world_state_t100_diagnostic = 0.22857426649949408
full_trajectory_world_state_hard = 0.19518047277951456
full_trajectory_world_state_easy = 0.0
full_trajectory_world_state_positive_domains = 2
full_trajectory_world_state_interaction_auroc = 0.9614642176190807
full_trajectory_world_state_occupancy_auroc = 0.9486653948303418
goal_route_physical_pass = True
goal_route_top1 = 0.7590404840801037
goal_route_majority_top1 = 0.5532884310618067
goal_route_lift_over_majority = 0.20575205301829702
physical_challenge_auroc = 0.9523668831032517
physical_challenge_auprc = 0.9931913407537012
physical_challenge_positive_rate = 0.8778634202564471
route_physical_policy_best_mode = no_route_physical
route_physical_policy_contributes = False
route_physical_policy_all_delta = 0.0
route_physical_policy_t50_delta = 0.0
route_physical_policy_hard_delta = 0.0
route_physical_group_deployable = True
route_physical_group_contributes = False
route_physical_group_all = 0.11568469607260612
route_physical_group_t50 = 0.08187483697323228
route_physical_group_t100_diagnostic = 0.12974873518015262
route_physical_group_hard = 0.12064147528200675
route_physical_group_easy = 0.0
route_physical_group_collision_delta_005 = 0.00585127847683331
route_physical_group_all_delta_vs_group = -0.10671970569760825
route_physical_group_t50_delta_vs_group = -0.06906290279898042
route_physical_group_hard_delta_vs_group = -0.10347621837197896
joint_route_conditioned_best = joint_route_conditioned_ensemble
joint_route_conditioning_contributes = False
joint_route_conditioned_all = 0.15088774687015682
joint_route_conditioned_t50 = 0.09295551695088011
joint_route_conditioned_t100_diagnostic = 0.16374924097478616
joint_route_conditioned_hard = 0.15655007967571088
joint_route_conditioned_all_delta_vs_full_traj = -0.03489077742818736
joint_route_conditioned_t50_delta_vs_full_traj = -0.05508147882643466
joint_multiagent_consistency_contributes = True
joint_multiagent_consistency_all = 0.18619055634397086
joint_multiagent_consistency_t50 = 0.14841646500755878
joint_multiagent_consistency_t100_diagnostic = 0.22857473063742806
joint_multiagent_consistency_hard = 0.19563212885213843
joint_multiagent_consistency_easy = 0.0
joint_multiagent_consistency_all_delta_vs_full_traj = 0.0004120320456266757
joint_multiagent_consistency_t50_delta_vs_full_traj = 0.0003794692302440117
joint_multiagent_consistency_expanded_on = 118
joint_policy_distillation_best = joint_distill_nobase_balanced::distiller_only
joint_policy_distillation_contributes = True
joint_policy_distillation_all = 0.28592959855458044
joint_policy_distillation_t50 = 0.21383787591021597
joint_policy_distillation_t100_diagnostic = 0.2887528737231674
joint_policy_distillation_hard = 0.28678460411829854
joint_policy_distillation_easy = 0.0
joint_policy_distillation_switch_rate = 0.42232747442731594
joint_policy_distillation_positive_domains = 2
joint_policy_distillation_all_delta_vs_joint_consistency = 0.09973904221060959
joint_policy_distillation_t50_delta_vs_joint_consistency = 0.06542141090265718
joint_policy_distillation_base_switch_input = False
joint_policy_distillation_bootstrap_all_low = 0.2816879475496606
joint_policy_distillation_bootstrap_t50_low = 0.20700457337312783
joint_policy_distillation_bootstrap_hard_low = 0.2822097538327134
joint_policy_distillation_stable = True
joint_policy_distillation_static_ablation_all_delta = -0.17365892957312368
joint_policy_distillation_prediction_ablation_all_delta = -0.008307576365714331
joint_policy_distillation_multiseed_pass = True
joint_policy_distillation_multiseed_all_mean = 0.2855577482364627
joint_policy_distillation_multiseed_all_min = 0.2785936928672702
joint_policy_distillation_multiseed_t50_mean = 0.19436183988319766
joint_policy_distillation_multiseed_t50_min = 0.1695617463193938
joint_policy_distillation_multiseed_easy_max = 0.0
ucy_fallback_repair_contributes = True
ucy_fallback_repair_all = 0.3613141132176878
ucy_fallback_repair_t50 = 0.25956635248380133
ucy_fallback_repair_t100_diagnostic = 0.37474907455985007
ucy_fallback_repair_hard = 0.3616933168487243
ucy_fallback_repair_easy = 0.0
ucy_fallback_repair_ucy_all = 0.3928657400363359
ucy_fallback_repair_ucy_t50 = 0.24265047375057225
ucy_fallback_repair_bootstrap_ucy_low = 0.38373376338122456
ucy_internal_validation_pass = True
ucy_source_level_validation_available = False
ucy_source_level_blocker = UCY has one train source and no UCY validation source; true source-level UCY validation needs another UCY-like source or a rebuilt split.
joint_rollout_consistency_pass = True
joint_rollout_consistency_all = 0.20681231782543796
joint_rollout_consistency_t50 = 0.141381936033941
joint_rollout_consistency_t100_diagnostic = 0.1391835100380775
joint_rollout_consistency_hard = 0.20016282886383174
joint_rollout_consistency_easy = 0.0
joint_rollout_consistency_multi_agent_all = 0.20472359322136924
joint_rollout_consistency_collision_delta_005 = -0.004879122491654175
joint_latent_rollout_deployable = False
joint_latent_rollout_improves_current = False
joint_latent_rollout_all = 0.0
joint_latent_rollout_t50 = 0.0
joint_latent_rollout_t100_diagnostic = 0.0
joint_latent_rollout_hard = 0.0
joint_latent_rollout_easy = 0.0
joint_latent_raw_neural_all = -0.7185960593471277
joint_latent_raw_neural_t50 = -0.9470536951712247
joint_latent_raw_neural_easy = 7.0398942498342745
joint_latent_interaction_auroc = 0.9778962684138552
joint_latent_occupancy_auroc = 0.9533724454671079
joint_latent_future_group_close_auroc = 0.9718721574984199
joint_residual_rollout_selected_trial = joint_residual_clip050_safe
joint_residual_rollout_deployable = False
joint_residual_rollout_improves_current = False
joint_residual_rollout_all = -0.006199871934993384
joint_residual_rollout_t50 = -0.01622586160854622
joint_residual_rollout_t100_diagnostic = -0.0055173545259843415
joint_residual_rollout_hard = -0.0070840559248055435
joint_residual_rollout_easy = 0.013581362841401212
joint_residual_raw_neural_all = -0.3187259758636325
joint_residual_raw_neural_t50 = -0.4973167857250549
joint_residual_raw_neural_easy = 3.0795776161376986
joint_residual_interaction_auroc = 0.9656852889954971
joint_residual_occupancy_auroc = 0.9383229818070397
joint_residual_future_group_close_auroc = 0.9677306100575066
joint_residual_domain_policy_selected_trial = joint_residual_clip100_balanced
joint_residual_domain_policy_deployable = False
joint_residual_domain_policy_all = -0.0006363835790781369
joint_residual_domain_policy_t50 = 0.0
joint_residual_domain_policy_t100_diagnostic = -0.0005775775254206472
joint_residual_domain_policy_hard = -0.0005725919532908463
joint_residual_domain_policy_easy = 0.00111839385467416
joint_residual_domain_policy_switch_rate = 0.0020530182970753493
all_agent_composite_world_state_pass = True
all_agent_composite_rows = 55528
all_agent_composite_ade_all = 0.2102513255185352
all_agent_composite_ade_t50 = 0.13652231450154184
all_agent_composite_ade_t100_diagnostic = 0.14694086716388166
all_agent_composite_ade_hard = 0.20384916307933942
all_agent_composite_fde_all = 0.19816955620782206
all_agent_composite_fde_t50 = 0.17387876491801468
all_agent_composite_multi_agent_ade_all = 0.20822625789447657
all_agent_composite_multi_agent_ade_t50 = 0.1379658977168079
all_agent_composite_collision_delta_005 = -0.0038702813749587617
teacher_guided_proposal_selected_trial = teacher_proposal_balanced
teacher_guided_proposal_deployable_raw = False
teacher_guided_proposal_all = 0.35147372419646705
teacher_guided_proposal_t50 = 0.23666446707361
teacher_guided_proposal_t100_diagnostic = 0.3579659237738704
teacher_guided_proposal_hard = 0.350941376256631
teacher_guided_proposal_easy = 0.0
teacher_guided_proposal_collision_delta_005 = 0.018672731941744014
teacher_guided_repair_deployable = True
teacher_guided_repair_improves_current = True
teacher_guided_repair_all = 0.20359710771827477
teacher_guided_repair_t50 = 0.13116399043122728
teacher_guided_repair_t100_diagnostic = 0.13371172832175005
teacher_guided_repair_hard = 0.19657225579495552
teacher_guided_repair_easy = 0.0
teacher_guided_repair_switch_rate = 0.2954185275896845
teacher_guided_repair_collision_delta_005 = -0.003961994203749264
teacher_guided_repair_all_delta_vs_current = 0.06371719082858676
teacher_guided_repair_t50_delta_vs_current = 0.009665968695006091
teacher_guided_repair_t100_delta_vs_current = -0.03520915160486934
teacher_guided_repair_hard_delta_vs_current = 0.05152997543259538
teacher_guided_evidence_pass = True
teacher_guided_bootstrap_all_low = 0.19991125433418688
teacher_guided_bootstrap_t50_low = 0.12503056367179802
teacher_guided_bootstrap_t100_low = 0.1270625048070677
teacher_guided_bootstrap_hard_low = 0.19259794747674494
teacher_guided_bootstrap_domain_lows = 0.17604451654489478 / 0.21892619124157922 / 0.22065013211805656
teacher_guided_no_fallback_all = 0.296621240422128
teacher_guided_no_fallback_easy = 1.2458611044726973
teacher_guided_no_group_consistency_all_delta = -0.1993508302918573
teacher_guided_no_neighbor_interaction_all_delta = -0.1980910960572272
teacher_guided_multiseed_replication_pass = True
teacher_guided_multiseed_all_mean = 0.20399416929662803
teacher_guided_multiseed_all_min = 0.20358992224459205
teacher_guided_multiseed_t50_mean = 0.13176918009378483
teacher_guided_multiseed_t50_min = 0.13019734119222148
teacher_guided_multiseed_t100_mean = 0.1349446267607578
teacher_guided_multiseed_t100_min = 0.13366944267790382
teacher_guided_multiseed_hard_mean = 0.1970419557530916
teacher_guided_multiseed_hard_min = 0.19653792868736553
teacher_guided_multiseed_easy_max = 0.0
teacher_guided_multiseed_collision_delta_max = -0.0037418834146520363
teacher_guided_multiseed_positive_domain_counts = [3, 3, 3]
composite_tail_evidence_pass = True
composite_tail_strict_delta_vs_teacher_pass = True
composite_tail_all = 0.2102513255185352
composite_tail_t50 = 0.13652231450154184
composite_tail_t100_diagnostic = 0.14694086716388166
composite_tail_hard = 0.20384916307933942
composite_tail_easy = 0.0
composite_tail_switch_rate = 0.3410171445036738
composite_tail_collision_delta_005 = -0.0038702813749587617
composite_tail_bootstrap_lows_all_t50_t100_hard = 0.20671347297933704 / 0.13060829691569112 / 0.13962817164239194 / 0.19986489207982036
composite_tail_delta_vs_teacher_lows_all_t50_t100_hard = 0.006356558489780059 / 0.004948699786380758 / 0.01249062196636364 / 0.006931417129331407
composite_tail_multiseed_pass = True
composite_tail_multiseed_strict_delta_pass = True
composite_tail_multiseed_positive_domain_counts = [3, 3, 3]
pure_ucy_source_heldout_gate = True
pure_ucy_three_way_train_val_test_gate = False
pure_ucy_policy_train_val_test_gate = True
strict_pure_ucy_only_neural_retrain_gate = True
strict_pure_ucy_neural_best_trial = pure_ucy_transformer
strict_pure_ucy_neural_best_mode = bounded_endpoint_residual
strict_pure_ucy_neural_all = 0.09008338741058997
strict_pure_ucy_neural_t50 = 0.08800918427966675
strict_pure_ucy_neural_t100_diagnostic = 0.08310612104619552
strict_pure_ucy_neural_hard = 0.09358425348430643
strict_pure_ucy_neural_easy = 0.0
strict_pure_ucy_neural_blocker = 
strict_pure_ucy_neural_statistical_evidence_pass = True
strict_pure_ucy_neural_bootstrap_lows_all_t50_t100_hard = 0.08888491295965614 / 0.08630755189944371 / 0.08069804446646152 / 0.09229635491900144
domain_local_endpoint_two_domain_gate = True
domain_local_endpoint_positive_domains = ['ETH_UCY', 'TrajNet', 'UCY_expanded']
domain_local_all_agent_two_domain_gate = True
domain_local_all_agent_positive_domains = ['ETH_UCY', 'UCY_expanded']
domain_local_full_waypoint_two_domain_gate = False
domain_local_full_waypoint_positive_domains = []
domain_local_full_waypoint_failure_taxonomy = {'ETH_UCY': {'reasons': ['t50_ade_not_positive', 't100_ade_not_positive', 'same_frame_proximity_delta_unsafe', 'ungated_neural_catastrophic_easy_degradation'], 'ade_all': 0.03200185360974939, 'ade_t50': 0.0, 'ade_t100': 0.0, 'fde_all': 0.02140015213054336, 'fde_t50': 0.0, 'collision_delta_vs_floor_005': 0.020563149900689304, 'next_fix': 'train a validation-selected efficient proximity guard and horizon-specific waypoint policy; do not deploy learned full-waypoint domain-local model yet.'}, 'TrajNet': {'reasons': ['all_ade_not_positive', 't50_ade_not_positive', 'hard_failure_ade_not_positive', 'same_frame_proximity_delta_unsafe', 'ungated_neural_catastrophic_easy_degradation', 'endpoint_fde_positive_but_waypoint_ade_negative'], 'ade_all': -0.009876103097439914, 'ade_t50': -0.09471803973274517, 'ade_t100': 0.022840260699697912, 'fde_all': 0.06300366873346075, 'fde_t50': 0.06502418472340132, 'collision_delta_vs_floor_005': 0.014909478168264101, 'next_fix': 'train a validation-selected efficient proximity guard and horizon-specific waypoint policy; do not deploy learned full-waypoint domain-local model yet.'}}
endpoint_to_full_bridge_two_domain_gate = True
endpoint_to_full_bridge_positive_domains = ['ETH_UCY', 'TrajNet']
endpoint_to_full_bridge_claim = Endpoint neural dynamics are projected through a linear waypoint bridge and evaluated on actual reconstructed future waypoints. This is not learned waypoint-shape dynamics.
endpoint_to_full_statistical_gate = True
endpoint_to_full_statistical_positive_domains = ['ETH_UCY', 'TrajNet']
endpoint_to_full_statistical_domain_lows = {'ETH_UCY': {'gate': True, 'ade_all_low': 0.014989124720870283, 'ade_t50_low': 0.001358729620311877, 'ade_t100_low': 0.003205122362247967, 'ade_hard_low': 0.01477184652828257, 'ade_multi_low': 0.015119963379165813, 'fde_all_low': 0.01541917336054947, 'fde_t50_low': 0.0020177568013793916}, 'TrajNet': {'gate': True, 'ade_all_low': 0.03381106629559396, 'ade_t50_low': 0.018640699426599426, 'ade_t100_low': 0.007743473232062962, 'ade_hard_low': 0.03437455061849066, 'ade_multi_low': 0.03594728364084391, 'fde_all_low': 0.03386172479233887, 'fde_t50_low': 0.025763665503977688}}
required_ablation_coverage_gate = True
required_ablation_cross_protocol_limitations = []
same_protocol_architecture_ablation_gate = True
same_protocol_best_protected_architecture = Stage41_fresh_self_gated_endpoint_candidate
same_protocol_transformer_only_deployable = False
same_protocol_jepa_only_deployable = False
same_protocol_hybrid_deployable = False
learned_shape_calibrated_meta_gate = True
learned_shape_positive_domains = ['ETH_UCY', 'TrajNet']
learned_shape_claim = Learned waypoint-shape residual contribution is positive but small and protected by endpoint bridge/floor fallback; it is not an ungated full-row neural replacement.
group_consistency_distiller_deployable = True
group_consistency_distiller_improves_fixed_guard = True
group_consistency_distiller_all = 0.22240440177021437
group_consistency_distiller_t50 = 0.1509377397722127
group_consistency_distiller_t100_diagnostic = 0.23019369783249866
group_consistency_distiller_hard = 0.2241176936539857
group_consistency_distiller_easy = 0.0
group_consistency_distiller_collision_delta_005 = 0.00829083972266037
group_consistency_distiller_t100_delta_vs_fixed_guard = 0.09101018779442116
group_consistency_distiller_bootstrap_all_low = 0.2185104674424955
group_consistency_distiller_bootstrap_t50_low = 0.1445060231000635
group_consistency_distiller_bootstrap_t100_low = 0.22247098030108228
group_consistency_distiller_bootstrap_hard_low = 0.2197743039844412
group_consistency_distiller_stable = True
group_consistency_distiller_group_feature_ablation_all_delta = -0.22219845243926872
group_consistency_distiller_proposal_score_ablation_all_delta = -0.22195167063440968
group_consistency_multiseed_initial_pass = False
group_consistency_multiseed_safety_buffer_pass = True
group_consistency_multiseed_all_mean = 0.139879916889688
group_consistency_multiseed_all_min = 0.11979149779858422
group_consistency_multiseed_t50_mean = 0.12149802173622119
group_consistency_multiseed_t50_min = 0.11587051220062827
group_consistency_multiseed_t100_mean = 0.1689208799266194
group_consistency_multiseed_t100_min = 0.13813952571953958
group_consistency_multiseed_hard_mean = 0.14504228036236014
group_consistency_multiseed_hard_min = 0.12488209249854709
group_consistency_multiseed_easy_max = 0.0
group_consistency_multiseed_collision_delta_max = 0.008309182288418482
group_consistency_multiseed_positive_domain_counts = [3, 3, 3]
jepa_deployment_decision = disable_jepa_in_deployable_path
jepa_disable_deployable_path = True
jepa_attempt_count = 7
jepa_non_collapse_attempt_count = 7
jepa_deployable_positive_attempt_count = 0
stage5c_executed = false
smc_enabled = false
```

Next target: post-Stage41 strengthening should pursue larger independent external data, safer ungated/full-row neural dynamics, and metric/time calibration. Current claims remain protected dataset-local raw-frame 2.5D, not true 3D or foundation.
<!-- M3W_NEURAL_COMPLETION_AUDIT:END -->

## Stage41 Source-Level Validation Repair

Fresh source-level audit of the frozen teacher-guided candidate was added under `outputs/stage41_external_split/stage41_source_level_validation_repair.md`.

```text
source = fresh_run
source_level_validation_repair_pass = true
overall_all_improvement = 0.20359710771827477
overall_t50_improvement = 0.13116399043122728
overall_t100_raw_frame_diagnostic = 0.13371172832175005
overall_hard_failure_improvement = 0.19657225579495552
overall_easy_degradation = 0.0
positive_heldout_sources = 3
pure_ucy_source_level_gate = false
ucy_family_surrogate_gate = true
stage5c_executed = false
smc_enabled = false
```

Interpretation: the frozen teacher-guided neural candidate is positive on held-out source files and on a UCY-family surrogate, but pure UCY source-level validation remains blocked because the available split has no independent UCY validation source after excluding duplicate-like zara03. This supports candidate status, not final foundation or true-3D claims.

## Stage41 Pure UCY Source-Heldout Validation

Fresh pure-UCY source-heldout audit was added under `outputs/stage41_external_split/stage41_pure_ucy_source_validation.md`. The composite-tail policy was selected only on non-UCY validation rows and then evaluated once on UCY held-out sources, without future endpoints, central velocity, test endpoint goals, Stage5C, or SMC.

```text
source = fresh_run
policy_selected_on = non_ucy_validation_rows_only
pure_ucy_source_heldout_gate = true
pure_ucy_three_way_train_val_test_gate = false
UCY/zara01 all = 0.2183, t50 = 0.1305, t100 = 0.1520, hard = 0.2151, easy = 0.0000
UCY/zara02 all = 0.1906, t50 = 0.1358, t100 = 0.1346, hard = 0.1871, easy = 0.0000
UCY/zara03 all = 0.2327, t50 = 0.0914, t100 = 0.1926, hard = 0.2260, easy = 0.0000
stage5c_executed = false
smc_enabled = false
```

Interpretation: this repairs the narrower pure-UCY held-out evidence gap for the frozen-policy check, but it is still not a pure UCY-only retrain/select/test protocol because the frozen model and safety floor were trained on mixed external train data. Dataset-local raw-frame claims only.

## Stage41 Bounded Neural Blend Dynamics

Fresh bounded blend evaluation tested whether a continuous neural dynamics head `floor + alpha * (neural - floor)` can contribute without binary full replacement.

```text
source = fresh_run
selected_policy = global alpha 0.3
deployable = false
all_improvement = 0.183054549856548
t50_improvement = 0.17556642701259895
t100_raw_frame_diagnostic = 0.1988123052757771
hard_failure_improvement = 0.1934724604647473
easy_degradation = 0.2070880438160938
collision_delta_vs_floor_005 = 0.007905645841740305
stage5c_executed = false
smc_enabled = false
```

Interpretation: the continuous neural dynamics signal is real on all/t50/t100/hard, but the full-row easy-case harm is far beyond the <=2% safety gate. Full-row blend is not deployable. A second safe-switch family, constrained by the already validation-repaired teacher switch set and then allowing a small low-risk tail blend, is deployable as a bounded neural dynamics candidate:

```text
safe_switch_policy = composite_tail, switch_alpha 1.0, tail_alpha 0.08
safe_switch_deployable = true
safe_switch_all_improvement = 0.2102513255185352
safe_switch_t50_improvement = 0.13652231450154184
safe_switch_t100_raw_frame_diagnostic = 0.14694086716388166
safe_switch_hard_failure_improvement = 0.20384916307933942
safe_switch_easy_degradation = 0.0
safe_switch_collision_delta_vs_floor_005 = -0.0038702813749587617
safe_switch_delta_vs_teacher_repair_all = 0.006654217800260431
safe_switch_delta_vs_teacher_repair_t50 = 0.005358324070314557
safe_switch_delta_vs_teacher_repair_hard = 0.0072769072843839044
```

This proves a safe nonzero continuous neural-dynamics contribution exists under the Stage37/teacher safety floor and gives a small fresh-run lift over the teacher-guided repaired switch on all/t50/t100/hard while keeping easy degradation at 0.0. Follow-up bootstrap evidence for the frozen composite-tail policy is now positive, including positive CI lows for all/t50/t100/hard and positive delta-vs-teacher CI lows:

```text
composite_tail_evidence_pass = true
composite_tail_strict_delta_vs_teacher_pass = true
bootstrap_all_low = 0.20671347297933704
bootstrap_t50_low = 0.13060829691569112
bootstrap_t100_raw_frame_low = 0.13962817164239194
bootstrap_hard_low = 0.19986489207982036
delta_vs_teacher_all_low = 0.006356558489780059
delta_vs_teacher_t50_low = 0.004948699786380758
delta_vs_teacher_t100_low = 0.01249062196636364
delta_vs_teacher_hard_low = 0.006931417129331407
```

Composite-tail is now the strongest bootstrap-supported fresh candidate. A seed-aware follow-up reused the three independently trained teacher-guided seed checkpoints, selected composite-tail policies on each seed's validation split, and evaluated test once:

```text
composite_tail_multiseed_pass = true
composite_tail_multiseed_all_mean = 0.20954401723273208
composite_tail_multiseed_t50_mean = 0.1383020020634588
composite_tail_multiseed_t100_raw_frame_mean = 0.1445226429961963
composite_tail_multiseed_hard_mean = 0.203088119625216
composite_tail_multiseed_easy_max = 0.0
composite_tail_multiseed_delta_vs_teacher_all_min = 0.004997329897068581
composite_tail_multiseed_delta_vs_teacher_t50_min = 0.005636676602763568
composite_tail_multiseed_delta_vs_teacher_t100_min = 0.008330490182942518
composite_tail_multiseed_delta_vs_teacher_hard_min = 0.0054317016626029835
positive_domain_counts = [3, 3, 3]
```

Composite-tail is now bootstrap-supported, multiseed-supported, pure-UCY source-heldout supported, and backed by a UCY-only policy-head train/val/test calibration. A strict pure-UCY neural retrain/select/test audit has now been attempted; it is not deployable because the source-only neural signal still fails the safety/deployability gate, so the mixed-external M3W-Neural v1 candidate plus Stage37/teacher floor remains the current deployable path.

## Stage41 Locked-v2 Fixed Policy Confirmation Audit

This audit freezes the domain-safe relaxed policy and re-evaluates it without threshold re-selection. It reports domain/source/scene stress slices and split overlap checks, but it is still not a fresh external dataset confirmation.

```text
source = fresh_run
deployment_decision = candidate_needs_fresh_external_confirmation_before_deployment
stage37_margin_pass = True
stress_pass = True
fresh_confirmation_pass = False
all_improvement = 0.17133358603743754
t50_improvement = 0.2368852639341944
t100_improvement = 0.19179621162648797
hard_failure_improvement = 0.1782452385355816
easy_degradation = 0.0
max_domain_easy_degradation = 0.0055509018258728116
```

Conclusion: fixed-policy stress evidence improved, but Stage37 remains the current deployable model until fresh external confirmation is completed.

## Stage41 Domain-Local Full-Trajectory Repair

After the learned full-waypoint domain-local audit failed, a fresh repair pass tested endpoint-linearized trajectory modes, a train-fitted gain-calibrated switch head, horizon-specific deployment variants, and a validation-selected proximity guard. This is still dataset-local raw-frame 2.5D evidence, not metric/seconds-level, not true 3D, and not Stage5C/SMC.

```text
source = fresh_run
positive_domains = ['TrajNet']
two_domain_repair_gate = False
ETH_UCY_all = 0.011436216574168045
ETH_UCY_t50 = 0.0
ETH_UCY_t100 = 0.027826091852716672
ETH_UCY_hard = 0.012286415999727573
ETH_UCY_easy = 0.030422051582583265
TrajNet_variant = t50_only
TrajNet_all = 0.002773658373583787
TrajNet_t50 = 0.010928773473788844
TrajNet_hard = 0.0031067454048834264
TrajNet_easy = 0.003963066360362033
TrajNet_collision_delta_005 = -0.00045641259698764314
pytest = 213 passed in 60.42s
```

Conclusion: the repair produced one deployable full-waypoint neural dynamics slice on TrajNet t50, but ETH_UCY still fails t50/easy and the two-domain neural world-dynamics gate remains false. Current best deployable remains the protected composite-tail/Stage37-floor route; Stage5C and SMC stay disabled.

## Stage41 Endpoint-To-Full-Trajectory Bridge

To test whether endpoint neural dynamics could repair the ETH_UCY full-waypoint t50 blocker, the domain-local endpoint model was projected into linear waypoint rollouts and scored against reconstructed actual future waypoint labels. The policy and proximity guard were selected on validation only; test was evaluated once. This is endpoint neural dynamics with a linear waypoint bridge, not learned full-waypoint shape dynamics, not metric/seconds-level, and not Stage5C/SMC.

```text
source = fresh_run
positive_domains = ['ETH_UCY', 'TrajNet']
two_domain_endpoint_to_full_gate = True
ETH_UCY_all = 0.01569515982437708
ETH_UCY_t50 = 0.001900902571733143
ETH_UCY_t100 = 0.004314304188029139
ETH_UCY_hard = 0.015473286009687448
ETH_UCY_easy = 0.0
ETH_UCY_collision_delta_005 = -0.0014959945951163456
TrajNet_all = 0.038025206221747654
TrajNet_t50 = 0.02647590218373508
TrajNet_t100 = 0.013800550192567762
TrajNet_hard = 0.03906048059123168
TrajNet_easy = 0.0
TrajNet_collision_delta_005 = -0.0029078220412910305
pytest = 215 passed in 59.77s
```

Conclusion: this is the first Stage41 domain-local bridge evidence where endpoint neural dynamics remain positive when evaluated as full future waypoint rollouts on two external domains. It strengthens the neural world-state case but still does not prove learned full-waypoint shape dynamics; the next step is to train a waypoint-shape model to match or exceed this bridge.

## Stage41 Learned Waypoint-Shape Bridge

After the endpoint-to-full linear bridge passed two external domains, a fresh learned waypoint-shape residual head was trained from past-only features and future waypoint labels. The residual head predicts shape corrections around the endpoint neural bridge; policy thresholds are validation-selected, test is evaluated once, and the deployment remains protected by the endpoint bridge/floor fallback. This is still dataset-local raw-frame 2.5D evidence, not metric/seconds-level, not true 3D, and not Stage5C/SMC.

```text
source = fresh_run
positive_domains = ['ETH_UCY', 'TrajNet']
two_domain_learned_shape_gate = True
ETH_UCY_all = 0.015700181822596138
ETH_UCY_t50 = 0.001900902571733143
ETH_UCY_t100 = 0.004328710539971414
ETH_UCY_hard = 0.015478633818655996
ETH_UCY_easy = 0.0
ETH_UCY_shape_gain_all_t50_t100_hard = 0.0000051020761192566155 / 0.0 / 0.000014468774637288462 / 0.00000543185765555787
ETH_UCY_shape_switch_rate = 0.00004630058338735068
ETH_UCY_collision_delta_005 = -0.0014959945951163456
TrajNet_all = 0.0382424875667583
TrajNet_t50 = 0.02647590218373508
TrajNet_t100 = 0.014508831312789572
TrajNet_hard = 0.039298907023759044
TrajNet_easy = 0.0
TrajNet_shape_gain_all_t50_t100_hard = 0.00022587010222718806 / 0.0 / 0.0007181925728715344 / 0.0002481180425111251
TrajNet_shape_switch_rate = 0.0010992030777686177
TrajNet_collision_delta_005 = -0.0029078220412910305
pytest = 217 passed in 60.50s
```

Conclusion: the learned shape head now passes a two-domain protected gate, but the actual learned-shape contribution is tiny and mostly t100/tail-specific. This is positive evidence that a learned waypoint-shape residual can be safely layered on top of the endpoint bridge, not proof of a large ungated full-waypoint neural dynamics breakthrough. Current claims remain protected 2.5D world-state evidence; Stage5C and SMC stay disabled.

## Stage41 Learned Shape Gain/Harm Gate

The next repair replaced the residual-norm heuristic with a train-fitted shape gain/harm gate. The gate is trained on train-only future-waypoint labels, selected on validation, and evaluated once on test. Inference remains past-only and protected by the endpoint bridge/floor fallback.

```text
source = fresh_run
positive_domains = ['ETH_UCY', 'TrajNet']
two_domain_gain_gate = True
ETH_UCY_all = 0.016363775908279754
ETH_UCY_t50 = 0.0017756136269108103
ETH_UCY_t100 = 0.0042703754565303065
ETH_UCY_hard = 0.016155508094891968
ETH_UCY_easy = 0.0
ETH_UCY_shape_gain_all_t50_t100_hard = 0.000679 / -0.000126 / -0.000044 / 0.000693
ETH_UCY_delta_vs_previous_shape_all_t100 = 0.000674 / -0.000059
TrajNet_all = 0.03813806330591063
TrajNet_t50 = 0.02693200074182789
TrajNet_t100 = 0.013800550192567762
TrajNet_hard = 0.03918432054611187
TrajNet_easy = 0.0
TrajNet_shape_gain_all_t50_t100_hard = 0.000117 / 0.000469 / 0.000000 / 0.000129
TrajNet_delta_vs_previous_shape_all_t100 = -0.000109 / -0.000718
pytest = 219 passed in 59.91s
```

Conclusion: learned gain/harm calibration expands ETH_UCY shape intervention and improves TrajNet t50 shape gain, but it is not a monotonic upgrade over the previous learned-shape bridge. The best next step is a domain/horizon-specific shape-policy composer that keeps ETH_UCY all/hard improvements from the gain gate while preserving TrajNet t100/tail behavior from the previous t100-only learned-shape policy. This remains protected 2.5D evidence; no Stage5C or SMC execution.

## Stage41 Domain/Horizon Shape-Policy Composer

A fresh composer now treats the previous learned-shape bridge and the gain/harm shape gate as separate sources. It selects the source per horizon family on validation only (`short`, `t50`, `t100`) and evaluates test once. The selected policy is still protected by the endpoint bridge/floor fallback and remains dataset-local raw-frame 2.5D evidence, not metric/seconds-level, true 3D, foundation-model evidence, Stage5C, or SMC.

```text
source = fresh_run
positive_domains = ['ETH_UCY', 'TrajNet']
two_domain_composer_gate = True
ETH_UCY_policy_short_t50_t100 = gain_gate / bridge / old_shape
ETH_UCY_all = 0.016413900592989306
ETH_UCY_t50 = 0.001900902571733143
ETH_UCY_t100 = 0.004328710539971414
ETH_UCY_hard = 0.016208884704508986
ETH_UCY_easy = 0.0
ETH_UCY_shape_gain_all_t50_t100_hard = 0.000730 / 0.000000 / 0.000014 / 0.000747
TrajNet_policy_short_t50_t100 = bridge / gain_gate / gain_gate
TrajNet_all = 0.03813806330591063
TrajNet_t50 = 0.02693200074182789
TrajNet_t100 = 0.013800550192567762
TrajNet_hard = 0.03918432054611187
TrajNet_easy = 0.0
TrajNet_shape_gain_all_t50_t100_hard = 0.000117 / 0.000469 / 0.000000 / 0.000129
pytest = 221 passed in 60.99s
```

Conclusion: the composer makes the mixed shape evidence deployable as a validation-selected horizon policy. It improves the ETH_UCY all/hard shape contribution while preserving the old-shape t100 behavior, and it keeps the TrajNet t50 gain-gate contribution. It does not yet prove a large ungated full-waypoint neural dynamics breakthrough; the protected floor remains required and Stage5C/SMC stay disabled.

## Stage41 Dynamic Shape Source Meta-Policy

The next experiment trained a per-row expected-ADE source model over three protected rollout sources: the endpoint bridge, the previous learned-shape bridge, and the gain/harm shape gate. The model used train future-waypoint labels only as supervision; validation selected gain/margin/source-rate thresholds; test was evaluated once. This tests whether source choice can be made dynamically from past-only causal features and candidate rollout geometry rather than from fixed horizon buckets.

```text
source = fresh_run
positive_domains = ['ETH_UCY', 'TrajNet']
two_domain_dynamic_meta_gate = True
ETH_UCY_all = 0.016314720529778892
ETH_UCY_t50 = 0.0017756136269108103
ETH_UCY_t100 = 0.004284781808472471
ETH_UCY_hard = 0.01611365182677149
ETH_UCY_easy = 0.0
ETH_UCY_shape_gain_all_t50_t100_hard = 0.000629 / -0.000126 / -0.000030 / 0.000650
ETH_UCY_source_distribution = bridge 0.9871 / old_shape 0.00005 / gain_gate 0.0128
ETH_UCY_test_ranking_accuracy = 0.0193
TrajNet_all = 0.03830169788706028
TrajNet_t50 = 0.02671519369046027
TrajNet_t100 = 0.014508831312789572
TrajNet_hard = 0.039363879492692044
TrajNet_easy = 0.0
TrajNet_shape_gain_all_t50_t100_hard = 0.000287 / 0.000246 / 0.000718 / 0.000316
TrajNet_source_distribution = bridge 0.9986 / old_shape 0.0011 / gain_gate 0.0003
TrajNet_test_ranking_accuracy = 0.9997
pytest = 224 passed in 60.12s
```

Conclusion: the dynamic meta-policy is safe and two-domain positive, and it improves TrajNet all/hard/t100 versus the fixed horizon composer. It does not dominate globally: ETH_UCY is weaker than the fixed composer and its ranking accuracy is poor, while TrajNet t50 is also slightly lower than the fixed composer. This is useful evidence for dynamic source selection, but the current best deployable path remains the protected composite/fixed-composer route until domain-specific calibration improves ETH_UCY ranking.

## Stage41 Calibrated Shape Source Meta-Policy

Because the dynamic meta-policy collapsed on ETH_UCY source ranking, the next experiment tested validation-only calibration of predicted source ADE. Four modes were compared: no calibration, global affine log calibration, source-specific calibration, and source+horizon calibration. Calibration uses validation source ADE labels only; test is evaluated once.

```text
source = fresh_run
positive_domains = ['ETH_UCY', 'TrajNet']
two_domain_calibrated_meta_gate = True
ETH_UCY_selected_calibration = none
ETH_UCY_all = 0.016314720529778892
ETH_UCY_t50 = 0.0017756136269108103
ETH_UCY_t100 = 0.004284781808472471
ETH_UCY_hard = 0.01611365182677149
ETH_UCY_easy = 0.0
ETH_UCY_rank_accuracy = 0.019261042689137885
ETH_UCY_delta_vs_fixed_all_t50_t100_hard = -0.000099 / -0.000125 / -0.000044 / -0.000095
TrajNet_selected_calibration = none
TrajNet_all = 0.03830169788706028
TrajNet_t50 = 0.02671519369046027
TrajNet_t100 = 0.014508831312789572
TrajNet_hard = 0.039363879492692044
TrajNet_easy = 0.0
TrajNet_rank_accuracy = 0.9997251992305578
TrajNet_delta_vs_fixed_all_t50_t100_hard = 0.000164 / -0.000217 / 0.000708 / 0.000180
pytest = 226 passed in 60.59s
```

Conclusion: simple affine calibration did not repair ETH_UCY; validation selected `none`, and source/source+horizon calibration made ranking worse. This negative result narrows the next fix: ETH_UCY needs pairwise gain/harm switch modeling or source-specific hard/failure features, not post-hoc raw-ADE calibration. Fixed horizon composer remains the safer deployable shape policy.

## Stage41 Pairwise Gain/Harm Shape Switch Policy

The follow-up repair trained a pairwise source-switch model that predicts gain and harm for switching from the protected bridge/Stage37 floor into the learned-shape or gain-gate source. This avoids the failed absolute-ADE ranking objective and selects conservative gain, harm, margin, and per-horizon switch-rate thresholds on validation only; test is evaluated once.

```text
source = fresh_run
positive_domains = ['ETH_UCY', 'TrajNet']
two_domain_pairwise_gate = True
domains_better_than_fixed_on_any_core_metric = ['TrajNet']
ETH_UCY_all = 0.01609092234292575
ETH_UCY_t50 = 0.0017756136269108103
ETH_UCY_t100 = 0.004284781808472471
ETH_UCY_hard = 0.015879478780563505
ETH_UCY_easy = 0.0
ETH_UCY_shape_gain_all_t50_t100_hard = 0.000402 / -0.000126 / -0.000030 / 0.000413
ETH_UCY_delta_vs_fixed_all_t50_t100_hard = -0.000323 / -0.000125 / -0.000044 / -0.000329
TrajNet_all = 0.038161213771437996
TrajNet_t50 = 0.02647590218373508
TrajNet_t100 = 0.014243899710901564
TrajNet_hard = 0.039209723936848406
TrajNet_easy = 0.0
TrajNet_shape_gain_all_t50_t100_hard = 0.000141 / 0.000000 / 0.000450 / 0.000155
TrajNet_delta_vs_fixed_all_t50_t100_hard = 0.000023 / -0.000456 / 0.000443 / 0.000025
pytest = 229 passed in 65.30s
```

Conclusion: pairwise gain/harm switching is safe and two-domain positive, but it still does not replace the fixed horizon composer. It repairs neither ETH_UCY’s weak ranking nor TrajNet t50; it only gives tiny TrajNet all/hard/t100 gains over fixed composer. Current best deployable shape policy remains the protected fixed composer/composite route, while pairwise switching is diagnostic evidence for future source-specific hard/failure features. Stage5C and SMC remain disabled.

## Stage41 Weighted Pairwise Shape Switch Policy

The next repair tried to address the rare-positive switch-label problem by upweighting hard/failure, t50/t100, source-switch, and positive-gain rows during pairwise gain/harm training. These labels are used only as training weights, never as inference inputs; validation still selects conservative safety thresholds and test is evaluated once.

```text
source = fresh_run
positive_domains = ['ETH_UCY', 'TrajNet']
two_domain_weighted_pairwise_gate = True
domains_better_than_fixed_on_any_core_metric = ['TrajNet']
ETH_UCY_all = 0.016058154598355467
ETH_UCY_t50 = 0.0017756136269108103
ETH_UCY_t100 = 0.004284781808472471
ETH_UCY_hard = 0.015850944660093846
ETH_UCY_easy = 0.0
ETH_UCY_shape_gain_all_t50_t100_hard = 0.000369 / -0.000126 / -0.000030 / 0.000384
ETH_UCY_delta_vs_fixed_all_t50_t100_hard = -0.000356 / -0.000125 / -0.000044 / -0.000358
TrajNet_all = 0.0382961343306194
TrajNet_t50 = 0.026692709235102585
TrajNet_t100 = 0.014508831312789572
TrajNet_hard = 0.039357774509706234
TrajNet_easy = 0.0
TrajNet_shape_gain_all_t50_t100_hard = 0.000282 / 0.000223 / 0.000718 / 0.000309
TrajNet_delta_vs_fixed_all_t50_t100_hard = 0.000158 / -0.000239 / 0.000708 / 0.000173
pytest = 231 passed in 66.04s
```

Conclusion: hard/tail/positive-gain weighting did not fix the remaining source-switch problem. It is safe and positive, but ETH_UCY remains below fixed composer and TrajNet t50 remains below fixed composer. The likely bottleneck is not simple positive-label imbalance; it is source-specific causal feature insufficiency or a need for stronger per-domain/t50 source-family priors. Fixed composer/composite remains the deployable route; Stage5C and SMC remain disabled.

## Stage41 Fixed-Composer Prior Source Switch Policy

The next repair treated the fixed horizon composer itself as the safety prior, then trained a residual source-switch model only around that prior. Validation was made stricter than the previous pairwise runs: a policy had to preserve t50 versus the fixed composer and improve at least one of all / hard / t100 on validation; otherwise the policy falls back to the fixed composer.

```text
source = fresh_run
positive_domains = ['ETH_UCY']
two_domain_fixed_prior_gate = False
domains_better_than_fixed_on_any_core_metric = []
two_domain_fixed_prior_beats_fixed_gate = False
ETH_UCY_all = 0.016339199386679604
ETH_UCY_t50 = 0.001900902571733143
ETH_UCY_t100 = 0.004284781808472471
ETH_UCY_hard = 0.01613176736923072
ETH_UCY_easy = 0.0
ETH_UCY_delta_vs_fixed_all_t50_t100_hard = -0.000075 / 0.000000 / -0.000044 / -0.000077
TrajNet_all = 0.03813806330591063
TrajNet_t50 = 0.02693200074182789
TrajNet_t100 = 0.013800550192567762
TrajNet_hard = 0.03918432054611187
TrajNet_easy = 0.0
TrajNet_delta_vs_fixed_all_t50_t100_hard = 0.000000 / 0.000000 / 0.000000 / 0.000000
pytest = 233 passed in 65.88s
```

Conclusion: the fixed-prior source switch is a clean negative result. The stricter validation rule prevented the TrajNet t50 harm seen in earlier dynamic selectors by reverting to the fixed composer, but it did not find a residual switch that beats the fixed composer on both domains. ETH_UCY still suffers a tiny all/hard/t100 loss versus fixed, and TrajNet is exactly the fixed composer. Current deployable shape policy remains the protected fixed composer/composite route. Stage5C and SMC remain disabled.

## Stage41 Fixed-Composer Residual Source Oracle Audit

After the fixed-prior learned switch failed, I measured the diagnostic oracle headroom for any per-row switch among `bridge`, `old_shape`, and `gain_gate` relative to the validation-selected fixed composer. This oracle uses future waypoint labels only to measure theoretical headroom; it is not an inference model and is not deployable.

```text
source = fresh_run
oracle_is_diagnostic_not_deployable = True
headroom_domains = []
two_domain_residual_oracle_headroom = False
ETH_UCY_oracle_delta_vs_fixed_all_t50_t100_hard = 0.000086 / 0.000013 / 0.000000 / 0.000073
ETH_UCY_positive_residual_rate = 0.001111
ETH_UCY_oracle_switch_rate = 0.760533
TrajNet_oracle_delta_vs_fixed_all_t50_t100_hard = 0.000244 / 0.000109 / 0.000708 / 0.000268
TrajNet_positive_residual_rate = 0.001374
TrajNet_oracle_switch_rate = 0.359714
pytest = 235 passed in 66.05s
```

Conclusion: the residual source-switch branch is nearly exhausted. The oracle switches often because of exact ties or near-zero margins, but truly positive residual rows are only about 0.1% of test rows, and absolute gains over the fixed composer are tiny. This explains why dynamic, calibrated, pairwise, weighted, and fixed-prior learned switches all failed to become deployable over the fixed composer. The next useful work is not another source-switch learner; it is stronger full-trajectory/group-world-state modeling or better external data/scene context. Stage5C and SMC remain disabled.

<!-- M3W_NEURAL_GOAL_COMPLETION:START -->
## M3W-Neural v1 Goal Completion Audit

The Stage41 breakthrough objective now has a requirement-by-requirement completion audit. It verifies the protected M3W-Neural v1 evidence package against the original Stage41 gates while keeping the claim boundaries explicit.

```text
goal_completion_status = complete
requirements_complete = 13 / 13
current_best_deployable = M3W-Neural v1 composite-tail safe-switch bounded neural dynamics candidate under Stage37/teacher floor (bootstrap+multiseed+pure-UCY source-heldout, UCY-only policy-head, and strict pure-UCY neural bootstrap evidence supported)
trained_neural_world_model = True
exceeds_stage37 = True
two_or_more_external_domains_positive = True
t50_improved = True
t100_improved_diagnostic = True
hard_failure_improved = True
easy_preserved = True
jepa_useful_for_deployable_path = False
foundation_world_model = False
stage5c_allowed = False
smc_allowed = False
```

This is still protected dataset-local/raw-frame 2.5D evidence, not true 3D, not metric/seconds-level, and not a foundation model. Stage5C and SMC remain disabled.
<!-- M3W_NEURAL_GOAL_COMPLETION:END -->

## Stage42-D Causal Ablation Evidence

```text
source = fresh_run
verdict = stage42_d_causal_ablation_evidence_pass_with_retrain_boundary
gates = 12 / 12
stage42_b_verdict = stage42_b_external_validation_pass_protected_neural_not_ungated
stage42_c_verdict = stage42_c_full_waypoint_dynamics_pass
required_ablation_coverage_gate = True
same_protocol_architecture_ablation_gate = True
protected_endpoint_all = 0.2102513255185352
protected_endpoint_t50 = 0.13652231450154184
protected_endpoint_hard_failure = 0.20384916307933942
protected_endpoint_easy_degradation = -0.14511076140795587
protected_full_waypoint_all = 0.18577852429834418
protected_full_waypoint_t50 = 0.14803699577731477
protected_full_waypoint_t100_raw_frame_diagnostic = 0.22857426649949408
protected_full_waypoint_hard_failure = 0.19518047277951456
protected_full_waypoint_easy_degradation = -0.0
all_components_retrained_inside_stage42_d = False
true_3d = false
foundation_world_model = false
metric_or_seconds_claim = false
stage5c_executed = false
smc_enabled = false
```

Stage42-D adds a causal ablation evidence package with strict source labels. Fresh rows recompute no-fallback, teacher-floor, endpoint-linear, and full-waypoint safety ablations from Stage42-B/C. Required no-history/no-neighbor/no-scene-goal/no-interaction/no-JEPA/no-Transformer/no-fallback coverage is cached-verified from Stage30/41 evidence; it is not falsely relabeled as new Stage42 retraining.

## Stage42-E Safety Floor Research

```text
source = fresh_run
verdict = stage42_e_safety_floor_research_pass
gates = 12 / 12
best_policy_family = current_composite_tail_policy
best_policy_source = cached_verified_policy_fresh_eval
best_all = 0.2102513255185352
best_t50 = 0.13652231450154184
best_t100_raw_frame_diagnostic = 0.14694086716388166
best_hard_failure = 0.20384916307933942
best_easy_degradation = 0.0
floor_necessity_conclusion = teacher_floor_required_for_current_deployment
ungated_endpoint_easy_degradation = 1.2458611044726973
true_3d = false
foundation_world_model = false
metric_or_seconds_claim = false
stage5c_executed = false
smc_enabled = false
```

Stage42-E studies whether the Stage37/teacher floor can be removed. It evaluates internal self-gates, uncertainty/harm/conformal gates, teacher-prob gates, and bounded residual blends with validation-only threshold selection. Ungated neural remains unsafe; any partial floor removal is limited to explicitly deployable gated families.

## Stage42-F Paper Evidence Package

```text
source = fresh_run
verdict = stage42_f_paper_package_complete_not_full_a_journal_ready
gates = 12 / 12
full_a_journal_ready = False
external_all = 0.2102513255185352
external_t50 = 0.13652231450154184
full_waypoint_ade_all = 0.18577852429834418
full_waypoint_ade_t50 = 0.14803699577731477
safety_floor_best_all = 0.2102513255185352
safety_floor_best_easy = 0.0
all_components_retrained_inside_stage42_d = False
metric_or_seconds_claim = false
stage5c_executed = false
smc_enabled = false
```

Stage42-F packages A-E into paper-ready artifacts under `outputs/stage42_long_research/`. It supports a protected raw-frame 2.5D external world-state manuscript package, but it is **not yet full A-journal ready** because metric/time calibration, all-component fresh retrained ablation, independent external expansion, and floor-free safety remain open.

## Stage42-G Retrained Ablation Phase1

```text
source = fresh_run
verdict = stage42_g_retrained_ablation_phase1_pass
gates = 11 / 11
full_all = 0.81221615514233
full_t50 = 0.8461508150001023
full_t100_raw_frame_diagnostic = 0.9527078250428334
full_hard_failure = 0.8458841094532804
full_easy_degradation = -0.8412751638465125
phase1_not_full_stage42_d_completion = true
stage5c_executed = false
smc_enabled = false
```

Stage42-G Phase1 freshly refits external expected-FDE selectors for the key causal feature/safety variants. It improves the ablation evidence beyond cached coverage, but it still does not complete all A-journal retrained ablations because JEPA/Transformer/full-waypoint-shape retraining remains explicitly `not_run_in_phase1`.

## Stage42-H Causal Sequence Ablation

```text
source = fresh_run
verdict = stage42_h_sequence_ablation_pass
gates = 10 / 10
sequence_full_all = 0.7784711241234431
sequence_full_t50 = 0.7833622318578909
sequence_full_hard_failure = 0.8080734180137877
sequence_full_easy_degradation = -0.768403531092173
history_t50_delta_full_minus_no_history = 0.457817280518282
stage5c_executed = false
smc_enabled = false
```

Stage42-H trains a causal temporal sequence encoder, not a flattened-history ridge selector. It answers whether history tokens help under a sequence model while keeping val-only safety selection and test-once evaluation. This is still dataset-local raw-frame 2.5D evidence and not Stage5C/SMC.

## Stage42-I Sequence-To-Full-Waypoint Dynamics

```text
source = fresh_run
verdict = stage42_i_sequence_full_waypoint_partial
gates = 10 / 11
sequence_waypoint_full_ade_all = -0.01055804004793807
sequence_waypoint_full_ade_t50 = -0.03208177658024658
sequence_waypoint_full_ade_hard_failure = -0.011590796127162406
sequence_waypoint_full_ade_easy_degradation = 0.0
history_ade_t50_delta_full_minus_no_history = 0.0040235141863109725
stage5c_executed = false
smc_enabled = false
```

Stage42-I connects causal sequence history to actual reconstructed full-waypoint ADE/FDE labels. It remains a protected dataset-local raw-frame 2.5D dynamics experiment, not metric/seconds-level prediction and not Stage5C/SMC.

The honest interpretation is partial/failure, not deployment success: the full static+sequence waypoint head is ADE-negative, while `sequence_waypoint_no_static_context` is positive on ADE all/t50/hard (`0.0115`, `0.0199`, `0.0129`) and FDE t50 (`0.0611`) with easy degradation `0.0`. Next repair target is static-gated/static-dropout sequence-to-waypoint dynamics.

## Stage42-J Static-Gated Full-Waypoint Repair

```text
source = cached_verified_checkpoints_fresh_static_gate_eval
verdict = stage42_j_static_gated_full_waypoint_pass
gates = 10 / 10
static_gated_ade_all = 0.036222114075724364
static_gated_ade_t50 = 0.036875348395170704
static_gated_ade_hard_failure = 0.03970549853881511
static_gated_ade_easy_degradation = 0.0
static_gated_fde_t50 = 0.11663789673246368
stage5c_executed = false
smc_enabled = false
```

Stage42-J uses cached-verified Stage42-I no-static/full-static checkpoints and performs a fresh validation-selected static expert gate. It tests whether static/context should be allowed per domain/horizon rather than forced globally. It remains dataset-local raw-frame 2.5D evidence and not Stage5C/SMC.

Static-gated interpretation: Stage42-J repairs the Stage42-I failure mode at policy level. It is not a new checkpoint training run, so the source is explicitly `cached_verified_checkpoints_fresh_static_gate_eval`. The next stronger evidence step is a fresh static-gated/static-dropout checkpoint trained with this rule baked into the model.

## Stage42-K Fresh Static-Gated Checkpoint Training

```text
source = fresh_run
verdict = stage42_k_fresh_static_gated_checkpoint_pass
gates = 9 / 9
fresh_static_gated_ade_all = 0.013627569336276476
fresh_static_gated_ade_t50 = -0.01222845312944624
fresh_static_gated_ade_t100_raw_frame_diagnostic = 0.015857871472793977
fresh_static_gated_ade_hard_failure = 0.014790997177165513
fresh_static_gated_ade_easy_degradation = 0.0
fresh_static_gated_fde_t50 = 0.03584067679165526
fresh_static_gate_mean_test = 0.12781384587287903
stage5c_executed = false
smc_enabled = false
```

Stage42-K trains the static gate/dropout idea directly into a fresh checkpoint over three seeds. It is a real fresh-run improvement over the failed Stage42-I full static+sequence head and preserves easy cases, with positive ADE all/hard and FDE all/t50.

The honest boundary is important: Stage42-K does not beat the Stage42-J policy-level static expert gate, and its ADE t50 mean is still negative. So Stage42-K is a successful fresh-checkpoint repair step, not the new best deployable full-waypoint policy. Stage42-J remains the strongest static-gated full-waypoint evidence for now.

## Stage42-L Horizon-Aware T50 Static-Gate Repair

```text
source = fresh_run
verdict = stage42_l_horizon_static_gate_repair_pass
gates = 11 / 11
horizon_static_gate_ade_all = 0.021866490467258453
horizon_static_gate_ade_t50 = 0.0020146201423274133
horizon_static_gate_ade_hard_failure = 0.02396933275296098
horizon_static_gate_ade_easy_degradation = 0.0
horizon_static_gate_fde_t50 = 0.05315292474994737
horizon_static_gate_t50_mean = 0.19026817878087363
stage5c_executed = false
smc_enabled = false
```

Stage42-L targets the Stage42-K t+50 ADE failure with horizon-conditioned static gating and t+50-weighted training/policy selection. It remains dataset-local raw-frame 2.5D evidence and not Stage5C/SMC.

Stage42-L interpretation: the targeted horizon-aware repair works relative to Stage42-K (`ADE t50` moves from `-0.0122` to `+0.0020`, and FDE t50 improves from `0.0358` to `0.0532`) while preserving easy cases. It still does not beat Stage42-J's policy-level static gate, so Stage42-J remains the strongest static-gated full-waypoint evidence and Stage42-L is the strongest fresh checkpoint in this static-gated branch.

## Stage42-M Policy-Distilled Static Gate Checkpoint

```text
source = fresh_run
verdict = stage42_m_policy_distilled_static_gate_partial
gates = 10 / 12
policy_distilled_ade_all = 0.016145179493171253
policy_distilled_ade_t50 = -0.001543676155626487
policy_distilled_ade_hard_failure = 0.017697818504874285
policy_distilled_ade_easy_degradation = 0.0
policy_distilled_fde_t50 = 0.07290641189728979
policy_distilled_t50_gate_mean = 0.18051626284917197
stage5c_executed = false
smc_enabled = false
```

Stage42-M distills Stage42-J's validation-selected domain/horizon static expert choices into a fresh checkpoint. It remains dataset-local raw-frame 2.5D evidence and not Stage5C/SMC.

Stage42-M interpretation: this is a partial/negative repair. It improves FDE t50 (`0.0729`) over Stage42-L (`0.0532`) and preserves easy cases, but ADE t50 is still negative (`-0.0015`) and it does not beat Stage42-L or Stage42-J on ADE all/t50/hard. The likely failure mode is that Stage42-J's teacher is a coarse domain/horizon alpha, not a row-level gain/harm teacher, so it increases static usage without learning when static/context is locally harmful for ADE.

## Stage42-N Row-Level Gain/Harm Static-Gate Distillation

```text
source = fresh_run
verdict = stage42_n_row_gain_static_gate_partial
gates = 11 / 13
row_gain_ade_all = 0.025023795590058923
row_gain_ade_t50 = -0.02781637207460134
row_gain_ade_hard_failure = 0.026922830068713138
row_gain_ade_easy_degradation = 0.0
row_gain_fde_t50 = 0.05545595532346274
row_gain_t50_gate_mean = 0.2575782686471939
stage5c_executed = false
smc_enabled = false
```

Stage42-N replaces Stage42-M's coarse domain/horizon alpha teacher with row-level train/val static gain, floor gain, harm, and switchability supervision. This run is a single-teacher-seed row-level pilot with cached train/val teacher targets for recoverability. It remains dataset-local raw-frame 2.5D evidence and not Stage5C/SMC.

Stage42-N interpretation: this is a partial/negative t+50 result. It improves ADE all (`0.0250`) and hard/failure (`0.0269`) over Stage42-L/M while preserving easy cases, but ADE t50 becomes negative (`-0.0278`). Row-level alpha supervision is therefore not sufficient; the next repair needs an explicit row-level gain/harm/switchability selector head or t+50-specific teacher ensemble rather than only a static-gate alpha target.

## Stage42-O Explicit Row-Level Gain/Harm Selector Head

```text
source = fresh_run
verdict = stage42_o_explicit_gain_harm_selector_partial
gates = 13 / 14
explicit_selector_ade_all = 0.0526457864037421
explicit_selector_ade_t50 = -0.0007755414586538093
explicit_selector_ade_hard_failure = 0.0535270529782426
explicit_selector_ade_easy_degradation = 0.015491233410829327
explicit_selector_fde_t50 = 0.05761440213671524
feature_normalization = train_split_stats_only
no_test_statistics_normalization = true
stage5c_executed = false
smc_enabled = false
```

Stage42-O adds an explicit row-level gain/harm/switchability selector head on top of cached-verified Stage42-N full-waypoint predictors. It remains dataset-local raw-frame 2.5D evidence and not Stage5C/SMC.

Stage42-O interpretation: after fixing normalization to use train-split statistics only, the result is a useful partial repair rather than a t+50 pass. It improves ADE all and hard/failure over Stage42-N and keeps easy degradation below the mean 2% gate, but ADE t50 remains slightly negative, so it must not be packaged as a t+50 success.

## Stage42-P T50-Specific Gain/Harm Selector Repair

```text
source = fresh_run
verdict = stage42_p_t50_gain_harm_selector_pass
gates = 14 / 14
t50_gain_harm_ade_all = 0.051537041008552574
t50_gain_harm_ade_t50 = 0.006595599081553938
t50_gain_harm_ade_hard_failure = 0.05325620637574713
t50_gain_harm_ade_easy_degradation = 0.008580272800932839
t50_gain_harm_fde_t50 = 0.057430632009597526
feature_normalization = train_split_stats_only
stage5c_executed = false
smc_enabled = false
```

Stage42-P is a t+50-specific follow-up to Stage42-O. It increases t+50 teacher weight and searches a t+50-weighted validation policy while preserving the raw-frame/dataset-local 2.5D claim boundary.

## Stage42-Q T50 Static Expert + Gain/Harm Combo

```text
source = cached_verified_report_level_preflight
verdict = stage42_q_preflight_partial_row_cache_required
gates = 7 / 7
diagnostic_ade_all_best_available = 0.0526457864037421
diagnostic_ade_t50_best_available = 0.036875348395170704
diagnostic_ade_hard_best_available = 0.0535270529782426
diagnostic_fde_t50_best_available = 0.11663789673246368
row_level_combo_status = attempted_not_completed
stage5c_executed = false
smc_enabled = false
```

Stage42-Q targets the complementarity between Stage42-J static-gated full-waypoint experts and Stage42-P t+50 gain/harm selector. If it is a preflight result, it is diagnostic only and not a deployable combo claim; a row-level NPZ prediction cache is required before a full validation-only combo can be treated as pipeline evidence.

## Stage42-R Row Prediction Cache + Combo Eval

```text
source = fresh_run_from_row_prediction_cache
verdict = stage42_r_row_cached_combo_pass
gates = 15 / 15
cached_combo_ade_all = 0.05238704221741153
cached_combo_ade_t50 = 0.03793420310086152
cached_combo_ade_t50_ci_low = 0.02774018469754745
cached_combo_ade_hard_failure = 0.05479172593908743
cached_combo_ade_easy_degradation = 0.001101978371627214
cached_combo_fde_t50 = 0.10005888767615174
cache_dir = data/stage42_row_prediction_cache (not committed)
stage5c_executed = false
smc_enabled = false
```

Stage42-R builds a local NPZ row prediction cache for floor / Stage42-J static expert / Stage42-P t+50 gain-harm selected errors, then performs validation-only combo evaluation from cache. It remains dataset-local raw-frame 2.5D evidence and not Stage5C/SMC.

## Stage42-S Frozen Row Combo Policy

```text
source = fresh_run_from_stage42r_row_cache
verdict = stage42_s_frozen_row_combo_policy_pass
gates = 13 / 13
policy_hash = 33450e033e14b10293b8a10796d934d7689e39358ab5eaa338d684a36b015d3f
cache_hash = f338f5c57b735b013ca210e30e9a6bbcfeebb646d4e0bc2e7f9e799006ac4ed6
ade_all = 0.05238704221741153
ade_t50 = 0.03793420310086152
ade_t50_ci_low = 0.02774018469754745
ade_hard_failure = 0.05479172593908743
ade_easy_degradation = 0.001101978371627214
stage5c_executed = false
smc_enabled = false
```

Stage42-S freezes the Stage42-R row-cache combo into a lightweight policy artifact and reports per-domain/per-horizon stress. It remains dataset-local raw-frame 2.5D evidence and not a metric, seconds-level, Stage5C, or SMC result.

## Stage42-T UCY Unseen-Domain Transfer Attempt

```text
source = fresh_run
verdict = stage42_t_ucy_transfer_blocked_no_candidate_predictions
gates = 8 / 11
ucy_ade_all = 0.0
ucy_ade_t50 = 0.0
ucy_hard_failure = 0.0
ucy_easy_degradation = 0.0
available_nonfloor_source_for_ucy = False
stage5c_executed = false
smc_enabled = false
```

Stage42-T attempts a validation-only unseen-domain transfer rule for UCY. The current row cache has no non-floor Stage42-J/P UCY predictions, so UCY remains fallback-only; this is reported as a blocker, not as a success.

## Stage42-W Unified External Full-Waypoint Policy

```text
source = fresh_unified_from_cached_verified_stage42s_and_stage42v
verdict = stage42_w_unified_external_full_waypoint_policy_pass
gates = 16 / 16
policy_hash = a2439e23c0c2e3f7aa99efa8a84e42868ea52258394ce41339c96ee0a2ec910e
rows = 55528
weighted_ADE_all = 0.09933852091487605
weighted_ADE_t50 = 0.09399823177957682
weighted_ADE_hard_failure = 0.10486717627981672
weighted_easy_degradation = 0.002399712905777252
domains = ETH_UCY, TrajNet, UCY
stage5c_executed = false
smc_enabled = false
```

Stage42-W combines ETH_UCY/TrajNet from the frozen Stage42-S row-cache combo policy with the UCY-domain slice from Stage42-V strict pure-UCY full-waypoint candidate. It avoids double counting the Stage42-V ETH_UCY slice and explicitly records that a single merged row-cache artifact remains future work. Claims remain dataset-local raw-frame 2.5D, not metric or seconds-level.

## Stage42-X Unified Row-Level Full-Waypoint Cache

```text
source = fresh_run_from_stage42s_row_cache_and_stage42v_ucy_predictions
verdict = stage42_x_unified_row_level_full_waypoint_cache_pass
gates = 16 / 16
cache_hash = ffa31b2525fa1a10db356ac5b1ef78602e44bc6f065c63cfc05ac29083e08937
ADE_all = 0.0900136608879362
ADE_t50 = 0.06109367671246102
ADE_t50_seed_CI_low = 0.05367075264893123
ADE_t50_bootstrap_CI_low = 0.027880326844751835
ADE_hard_failure = 0.09374591375146946
ADE_easy_degradation = 0.001101978371627214
positive_domains = ['ETH_UCY', 'TrajNet', 'UCY']
stage5c_executed = false
smc_enabled = false
```

Stage42-X upgrades Stage42-W from a domain-level policy package into a row-level merged full-waypoint cache with unified bootstrap. ETH_UCY/TrajNet use Stage42-S row-cache combo outputs; UCY rows use Stage42-V UCY full-waypoint predictions after row alignment. Claims remain dataset-local raw-frame 2.5D, not metric or seconds-level.

## Stage42-Y Unified Ablation Evidence

```text
source = fresh_synthesis_from_stage42x_row_cache_and_retrained_ablation_reports
verdict = stage42_y_unified_ablation_evidence_pass
gates = 13 / 13
Stage42-X_ADE_all = 0.0900136608879362
Stage42-X_ADE_t50 = 0.06109367671246102
UCY_source_loss_if_removed_t50 = 0.0231594736115995
UCY_source_loss_if_removed_hard = 0.038954187812382024
history_token_t50_contribution = 0.457817280518282
history_token_hard_contribution = 0.47079873325328386
stage5c_executed = false
smc_enabled = false
```

Stage42-Y turns the Stage42-X unified row-level cache into paper-table ablation evidence. It shows that removing the UCY full-waypoint source loses t50/hard performance, history tokens are the strongest retrained sequence contribution, domain expert helps, and safety floor remains necessary because ungated neural is unsafe. Goal/scene and neighbor/interaction remain mixed rather than overclaimed.

Verification: `python3 run_stage42_unified_ablation_evidence.py` passed, `python3 -m pytest tests/test_stage42_unified_ablation_evidence.py` passed with 3 tests, and `python3 -m pytest tests` passed with 327 tests.

## Stage42-Z Paper Claim Evidence Audit

```text
source = fresh_audit_from_stage42_wxy_and_paper_package_artifacts
verdict = stage42_z_paper_claim_evidence_audit_pass
gates = 16 / 16
paper_ready_scope = protected_2p5d_raw_frame_world_state_candidate
not_ready_scope = true_3d_metric_seconds_foundation_or_stage5c_smc
supported_main_claims = unified row-level full-waypoint cache, positive t50 evidence, UCY source contribution, history-token/domain-expert contribution, protected external floor, protected full-waypoint sequence dynamics
mixed_or_non_claims = goal/scene uniform positivity, neighbor/interaction uniform positivity, ungated neural replacement, metric/seconds-level, true 3D/foundation
stage5c_executed = false
smc_enabled = false
```

Stage42-Z maps each paper claim to an explicit artifact and status. It confirms that Stage42 is paper-ready only as a protected dataset-local raw-frame 2.5D world-state candidate. It rejects ungated neural deployment, metric/seconds claims, true-3D/foundation claims, and overclaiming mixed goal/scene or neighbor/interaction evidence.

## Stage42-AA Retrained Ablation Matrix

```text
source = fresh_matrix_from_stage42g_rerun_plus_stage42h_i_d_z
verdict = stage42_aa_retrained_ablation_matrix_pass_with_jepa_transformer_boundary
gates = 15 / 15
fresh_required_coverage = 11 / 12
positive_fresh_contributions = no_history, no_neighbor, no_scene, no_goal, no_interaction, no_safe_switch, no_domain_expert
unsafe_negative_claim = no_teacher_floor
cached_not_overclaimed = no_JEPA
fresh_proxy_not_overclaimed = no_Transformer
stage5c_executed = false
smc_enabled = false
```

Stage42-AA reruns Stage42-G fresh retrained ablation and consolidates the required Stage42 ablations into one source-labeled matrix. It supports history tokens and domain expert as the clearest sequence contributions, shows goal/scene/neighbor/interaction are positive in the ridge retrained proxy but mixed at sequence level, and keeps JEPA/Transformer boundaries honest: no-JEPA is cached negative evidence, while no-Transformer is a fresh proxy rather than a full architecture retrain.

## Stage42-AB Full-Waypoint Auxiliary-Head Ablation

```text
source = fresh_run
verdict = stage42_ab_full_waypoint_auxiliary_ablation_pass
gates = 11 / 11
no_aux_ADE_all = -0.0023389398251364435
no_aux_ADE_t50 = -0.03744290181012914
no_aux_ADE_hard_failure = -0.0025638694532068573
no_aux_easy_degradation = 0.0
full_minus_no_aux_ADE_all = -0.008219100222801626
full_minus_no_aux_ADE_t50 = 0.005361125229882559
full_minus_no_aux_ADE_hard = -0.009026926673955549
uniform_aux_positive_claim_allowed = False
stage5c_executed = false
smc_enabled = false
```

Stage42-AB removes supervised interaction / occupancy / physical auxiliary losses while keeping the same full-waypoint model inputs, outputs, and validation-only policy interface. Positive deltas mean the auxiliary heads helped; mixed or negative deltas are recorded as limitation evidence, not overclaimed.

## Stage42-AC Paper Package Refresh

```text
source = fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts
verdict = stage42_ac_paper_package_refresh_pass
gates = 12 / 12
auxiliary_head_evidence = mixed_partial_not_uniform_main_claim
paper_ready_scope = protected_dataset_local_raw_frame_2p5d_world_state_candidate
stage5c_executed = false
smc_enabled = false
```

Stage42-AC refreshes the paper outline, method draft, experiment tables, ablation tables, failure taxonomy, model card, data card, reproducibility notes, and A-journal gap analysis with Stage42-AB. The auxiliary heads are now explicitly recorded as mixed evidence: small t50/FDE@50 support, but not uniform all/hard ADE improvement.

## Stage42-AK Post-Repair Locked Policy Audit

```text
source = fresh_synthesis_from_stage42_af_ag_ai_aj_and_source_split
verdict = stage42_ak_post_repair_locked_policy_audit_pass
gates = 17 / 17
policy_hash = 06772a241eedacc9b8828bddc7c70569ef7d0abc1951cc83eb1c5251e7979298
source_split_hash = e22c1fc43543da7fea1805460163f8fcd7993e3dcf88a2eb04d40a82269584bd
ade_all_ci_low = 0.0859783492681093
ade_t50_ci_low = 0.05851255877278698
ade_t100_raw_frame_diagnostic_ci_low = 0.06834922663403784
ade_hard_failure_ci_low = 0.0906618058871814
easy_degradation_ci_high = 0.00116827749002908
stage5c_executed = false
smc_enabled = false
```

Stage42-AK freezes the post-repair AF/AG/AI policy rules and source-level split audit as reproducibility evidence. It is a policy/source audit, not new training. Claims remain protected dataset-local raw-frame 2.5D; metric/seconds-level, true-3D, foundation, Stage5C, SMC, and ungated-neural deployment claims remain rejected.

## Stage42-AL Source-Level Coverage Audit

```text
source = fresh_synthesis_from_stage42_ak_ai_x_source_split
verdict = stage42_al_source_level_coverage_audit_pass_with_full_split_eval_gap
gates = 12 / 12
full_proposed_source_level_eval = false
ucy_source_test_coverage = exact_row_count_match
trajnet_source_test_coverage = partial_coverage
eth_ucy_stress_rows = extra_available_not_in_proposed_source_test
stage5c_executed = false
smc_enabled = false
```

Stage42-AL audits whether the locked post-repair policy can be claimed as a full proposed source-level split evaluation. It cannot: UCY matches the proposed source-level test row count, but TrajNet is only partially covered by the current locked-policy stress pool and ETH_UCY stress rows are extra available rows outside the proposed source-level test split. The correct claim remains available row-level post-repair stress with explicit coverage gap, not full source-level split evaluation.

## M3W Long-Term Execution Summary

```text
source = cached_verified_plus_stage42_ba_summary
file = README_M3W_EXECUTION_SUMMARY_ZH.md
scope = detailed route / failure / success / current best deployable summary
current_best_deployable = M3W-Neural v1 protected policy under Stage37 / teacher safety floor
current_claim = protected dataset-local raw-frame 2.5D multi-agent world-state candidate
t100_status = blocker / diagnostic after train-only source-CV guard
stage5c_executed = false
smc_enabled = false
```

A new reader-facing Chinese summary has been added at `README_M3W_EXECUTION_SUMMARY_ZH.md`. It consolidates what was attempted, which routes failed and why, which routes succeeded, the strongest meaningful metrics, and the strict non-claims: no true-3D, no foundation, no metric/seconds-level claim, no Stage5C execution, and no SMC.

## Stage42-BB T100 Data Gap Audit

```text
source = fresh_synthesis_from_stage42_ba_and_calibration
verdict = stage42_bb_t100_data_gap_audit_pass_with_data_blocker
gates = 14 / 14
unsupported_t100_domains = ETH_UCY, TrajNet, UCY
additional_t100_sources_needed = ETH_UCY:2, TrajNet:1, UCY:1
after_source_cv_guard_all = 0.280997
after_source_cv_guard_t50 = 0.289698
after_source_cv_guard_t100_raw_frame_diagnostic = 0.0
after_source_cv_guard_hard_failure = 0.251576
after_source_cv_guard_easy_degradation = -0.372431
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
stage5c_executed = false
smc_enabled = false
```

Stage42-BB turns the Stage42-BA t100 blocker into a concrete data/calibration action list. ETH_UCY has 0 safe-positive t100 source-CV folds and needs at least two additional safe t100-capable train sources or source-specific repair; TrajNet has 1 safe-positive fold and needs at least one; UCY lacks enough t100-capable original-train sources and needs at least one more. The protected all/t50/hard metrics remain positive after the source-CV guard, but t100 remains a raw-frame diagnostic blocker and must not be claimed as stable positive transfer.

Artifacts:

- `outputs/stage42_long_research/t100_data_gap_audit_stage42.md`
- `outputs/stage42_long_research/user_action_required_t100_stage42.md`
- `outputs/stage42_long_research/stage42_stage_bb_gate.md`

Verification: `python3 run_stage42_t100_data_gap_audit.py` passed, `python3 -m pytest tests/test_stage42_t100_data_gap_audit.py` passed with 4 tests, and `python3 -m pytest tests` passed with 426 tests.

## Stage42-BC T100 Source Acquisition Plan

```text
source = fresh_synthesis_from_stage42_bb_plus_official_web_pages
verdict = stage42_bc_t100_source_acquisition_plan_pass
gates = 11 / 11
candidate_sources = 6
official_sources_found = 5
local_path_found_sources = 6
high_priority_sources = ucy_crowd_original, trajnetpp_epfl_aicrowd, opentraj_toolkit, eth_ucy_original_sources
auto_download_allowed_sources = none
auto_download_executed = false
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
stage5c_executed = false
smc_enabled = false
```

Stage42-BC maps the Stage42-BB t100 source-support gap onto official-source and local-path actions. It uses official pages for TrajNet++ / VITA EPFL and DLR AerialMPT, plus local OpenTraj/SDD/UCY paths, and deliberately does not auto-download gated, terms-bound, or restricted-license raw data. The t100 repair priority is legal independent TrajNet++ / ETH-UCY / UCY source support followed by train-only source-CV reruns. AerialMPT is official and locally present but likely too short to solve t100 alone; SDD remains SDD-specific pixel raw-frame support, not an external t100 source-CV repair.

Artifacts:

- `outputs/stage42_long_research/t100_source_acquisition_plan_stage42.md`
- `outputs/stage42_long_research/user_action_required_t100_sources_stage42.md`
- `outputs/stage42_long_research/stage42_stage_bc_gate.md`

Verification: `python3 run_stage42_t100_source_acquisition_plan.py` passed, `python3 -m pytest tests/test_stage42_t100_source_acquisition_plan.py` passed with 4 tests, and `python3 -m pytest tests` passed with 430 tests.

## M3W Execution Summary Refresh Through Stage42-BT

```text
source = cached_verified_plus_stage42_bs_bt_summary_refresh
summary_file = README_M3W_EXECUTION_SUMMARY_ZH.md
current_best_deployable = M3W-Neural v1 protected policy under Stage37 / teacher safety floor
stage42_bs_verdict = stage42_bs_ucy_zara_t50_family_policy_pass_positive
stage42_bt_verdict = stage42_bt_eth_seq_t50_support_dry_run_pass_blocker_confirmed
stage5c_executed = false
smc_enabled = false
```

The reader-facing Chinese execution summary has been refreshed through Stage42-BT. It now includes the full route history, major success and failure reasons, Stage42-BS UCY_zara source-family-specific calibrated t50 repair, and Stage42-BT ETH_seq support dry-run blocker. UCY_zara now has a local positive calibrated t50 result (`t50 macro +24.72%`, min fold `+15.10%`, easy max `1.24%`), while ETH_seq remains blocked because ETH-Person XML technical h50 positives do not safely transfer to the `ETH_seq_eth` holdout and ETH-Person terms remain unverified. The summary keeps all claim boundaries explicit: no true-3D, no foundation, no global metric/seconds-level claim, no Stage5C execution, and no SMC.

Verification: `python3 run_stage42_ucy_zara_t50_family_policy.py` passed, `python3 run_stage42_eth_seq_t50_support_dry_run.py` passed, `python3 run_stage42_ucy_students_t50_source_support.py` passed, focused Stage42-BS/BT/BU tests passed with 7 tests, and `python3 -m pytest tests` passed with 497 tests.

## Stage42-BY Protected T50 Floor-Relaxability Repair

```text
source = fresh_stage42_by_t50_floor_relaxability_repair
verdict = stage42_by_t50_floor_relaxability_repair_pass
gates = 15 / 15
selected_variant = family_baseline_rel_only
internal_val_group = UCY::UCY/zara03/crowds_zara03.txt
repaired_t50_slices = TrajNet|50, UCY|50
global_t50_improvement = 28.97%
global_easy_degradation = -37.05%
TrajNet|50 t50 improvement = 30.21%
UCY|50 t50 improvement = 24.53%
floor_free_neural_deployable = false
teacher_floor_context_required = true
stage5c_executed = false
smc_enabled = false
```

Stage42-BY repairs the Stage42-BX t50 blockers under the Stage42-AW protected validation policy. `TrajNet|50` moved from `blocked_by_validation_safety` to protected positive, and `UCY|50` moved from `blocked_no_validation_support` to protected positive using train-only UCY internal validation support. This is deliberately not a floor-free neural deployment: the teacher/floor rollout context and protected fallback policy remain required. Claims remain dataset-local / raw-frame 2.5D; no true-3D, foundation, metric, seconds-level, Stage5C, or SMC claim is allowed.

Artifacts:

- `outputs/stage42_long_research/t50_floor_relaxability_repair_stage42.md`
- `outputs/stage42_long_research/t50_floor_relaxability_repair_stage42.json`
- `outputs/stage42_long_research/stage42_stage_by_gate.md`

Verification: `.venv-pytorch/bin/python run_stage42_t50_floor_relaxability_repair.py` passed and `python3 -m pytest tests/test_stage42_t50_floor_relaxability_repair.py tests/test_stage42_floor_relaxability_audit.py` passed with 9 tests.

## Stage42-BZ Protected T50 Repair Statistical Evidence

```text
source = fresh_stage42_bz_t50_repair_statistical_evidence
verdict = stage42_bz_t50_repair_statistical_evidence_pass
gates = 13 / 13
bootstrap_n = 3000
selected_variant = family_baseline_rel_only
internal_val_group = UCY::UCY/zara03/crowds_zara03.txt
robust_t50_slices = TrajNet|50, UCY|50
target_union_t50_improvement = 28.97%
target_union_t50_CI = [28.52%, 29.45%]
target_union_hard_failure_CI_low = 28.51%
target_union_easy_degradation_CI_high = -25.16%
TrajNet|50 t50_CI = [29.80%, 30.67%]
UCY|50 t50_CI = [23.02%, 26.08%]
floor_free_neural_deployable = false
stage5c_executed = false
smc_enabled = false
```

Stage42-BZ upgrades Stage42-BY from a protected t50 point-estimate repair to bootstrap-backed statistical evidence. The repaired `TrajNet|50` and `UCY|50` slices both have positive 3000-sample bootstrap lower bounds and easy degradation remains safely below the 2% harm limit. This remains protected policy evidence selected by train/internal-validation only; test rows are used only for final reporting/bootstrap. It is still not floor-free neural deployment, not true 3D, not foundation, and not metric/seconds-level.

Artifacts:

- `outputs/stage42_long_research/t50_repair_statistical_evidence_stage42.md`
- `outputs/stage42_long_research/t50_repair_statistical_evidence_stage42.json`
- `outputs/stage42_long_research/stage42_stage_bz_gate.md`

Verification: `.venv-pytorch/bin/python run_stage42_t50_repair_statistical_evidence.py` passed, focused Stage42-BY/BZ tests passed with 8 tests, and `python3 -m pytest tests` passed with 515 tests.

## Stage42-CA Post-BZ Paper Package Refresh

```text
source = fresh_synthesis_from_stage42_by_bz_artifacts
verdict = stage42_ca_post_bz_paper_package_refresh_pass
gates = 10 / 10
paper_files_refreshed = 9 / 9
included_evidence = Stage42-BY protected t50 repair + Stage42-BZ bootstrap evidence
target_union_t50_CI = [28.52%, 29.45%]
target_union_easy_degradation_CI_high = -25.16%
floor_free_neural_deployable = false
metric_seconds_claim = false
stage5c_executed = false
smc_enabled = false
```

Stage42-CA refreshes the paper package so the BY/BZ protected t50 repair evidence is visible in the outline, method draft, experiment tables, ablation tables, failure taxonomy, model card, data card, reproducibility note, and A-journal gap analysis. It does not train or tune anything. The refresh explicitly states that the result is protected policy evidence under the Stage37/teacher floor, not floor-free neural world dynamics, and it keeps raw-frame/dataset-local language.

Artifacts:

- `outputs/stage42_long_research/paper_package_post_bz_refresh_stage42.md`
- `outputs/stage42_long_research/paper_package_post_bz_refresh_stage42.json`
- `outputs/stage42_long_research/paper_package_post_bz_refresh_stage42.csv`
- `outputs/stage42_long_research/stage42_stage_ca_gate.md`

Verification: `.venv-pytorch/bin/python run_stage42_post_bz_paper_package_refresh.py` passed, focused CA/BZ tests passed with 7 tests, and `python3 -m pytest tests` passed with 518 tests.

## Stage42-CB Protected T50 Source Robustness Audit

```text
source = fresh_stage42_cb_t50_source_robustness_audit
verdict = stage42_cb_t50_source_robustness_pass_with_source_diversity_limit
gates = 11 / 11
robust_major_source_slices = TrajNet|50, UCY|50
concentration_limited_slices = TrajNet|50, UCY|50
broad_source_generalization_claim_allowed = false
TrajNet|50 sources = 2, largest_source_fraction = 99.08%
UCY|50 sources = 1, largest_source_fraction = 100.00%
stage5c_executed = false
smc_enabled = false
```

Stage42-CB audits whether the BY/BZ protected t50 gains are source-robust or concentrated. The major available sources are positive: `TrajNet|50` has robust evidence on `students003` and the small `students002` slice, and `UCY|50` is positive on `crowds_zara03`. However, the evidence is source-concentrated: `TrajNet|50` is 99.08% one source and `UCY|50` has only one test source. The correct paper claim is therefore major-source robustness within available rows, not broad source-level generalization. More independent legal t50-capable sources remain needed.

Artifacts:

- `outputs/stage42_long_research/t50_source_robustness_audit_stage42.md`
- `outputs/stage42_long_research/t50_source_robustness_audit_stage42.json`
- `outputs/stage42_long_research/stage42_stage_cb_gate.md`

Verification: `.venv-pytorch/bin/python run_stage42_t50_source_robustness_audit.py` passed, focused CB/BZ tests passed with 8 tests, and `python3 -m pytest tests` passed with 522 tests.

## Stage42-CC Independent T50 Source Inventory

```text
source = fresh_stage42_cc_independent_t50_source_inventory
verdict = stage42_cc_independent_t50_source_inventory_pass
gates = 10 / 10
scanned_files = 93
t50_capable_files = 10
unused_candidate_t50_sources = 0
alternate_current_source_candidates = 4
diagnostic_t50_candidates = 1
source_diversity_repair_ready = false
stage5c_executed = false
smc_enabled = false
```

Stage42-CC reruns the local t50 source inventory with conservative filtering after the Stage42-CB source-concentration warning. The result is intentionally stricter than the earlier stale inventory: no local file is currently counted as an unused independent ready-to-claim t50 source. `seq_eth/biwi_eth_10fps.txt`, `students01/students001.txt`, `students03/obsmat_px.txt`, and `students03/students003.txt` are treated as alternate representations or same-parent/current-source material useful for split rebuild diagnostics, not independent new held-out evidence. `synth_data/orca_circle_crossing_5ped.ndjson` is diagnostic/synthetic and cannot count as real external top-down pedestrian source support.

The correct conclusion is therefore: Stage42-BY/BZ t50 repair remains positive on available major sources, but broad source-level generalization is still blocked until a legal independent t50-capable top-down pedestrian source is converted, no-leakage audited, selected on train/internal-val, and tested once. Stage42-CC is an inventory and user-action audit, not conversion or benchmark success.

Artifacts:

- `outputs/stage42_long_research/independent_t50_source_inventory_stage42.md`
- `outputs/stage42_long_research/independent_t50_source_inventory_stage42.json`
- `outputs/stage42_long_research/stage42_stage_cc_gate.md`
- `outputs/stage42_long_research/user_action_required_independent_t50_sources_stage42.md`

Verification: `.venv-pytorch/bin/python run_stage42_independent_t50_source_inventory.py` passed, focused CC/CB tests passed with 10 tests, and `.venv-pytorch/bin/python -m pytest tests` passed with 528 tests. A prior `python3 -m pytest tests` run hit the known x86_64 Conda/OpenMP/subprocess crash path, so the authoritative full test result is the arm64 `.venv-pytorch` run.

## Stage42-CD Source Diversity Acquisition Package

```text
source = fresh_stage42_cd_source_diversity_acquisition_package
verdict = stage42_cd_source_diversity_acquisition_package_pass
gates = 13 / 13
official_targets = 5
critical_targets = 2
auto_download_targets = 0
manual_or_terms_targets = 4
local_paths_found = 4
converted_datasets_now = 0
source_diversity_repair_ready_now = false
broad_source_generalization_claim_allowed = false
stage5c_executed = false
smc_enabled = false
```

Stage42-CD turns the Stage42-CB/CC source-diversity blocker into an explicit official/manual acquisition package. It lists UCY Crowd Data, ETH/BIWI, TrajNet++, OpenTraj toolkit, and an additional legal top-down pedestrian/drone source target. Four local path families are present, but none are counted as converted datasets or as source-diversity repair. No auto-download was executed because the priority targets require manual terms/path verification or are toolkit/challenge references rather than independent data permission.

The correct conclusion remains conservative: BY/BZ protected t50 repair is paper-usable with source-concentration caveat, but broad source-level generalization is still blocked until a legal independent t50-capable source is provided, converted, no-leakage audited, selected on train/internal-val, and evaluated once on final test.

Artifacts:

- `outputs/stage42_long_research/source_diversity_acquisition_package_stage42.md`
- `outputs/stage42_long_research/source_diversity_acquisition_package_stage42.json`
- `outputs/stage42_long_research/stage42_stage_cd_gate.md`
- `outputs/stage42_long_research/user_action_required_source_diversity_stage42.md`

Verification: `.venv-pytorch/bin/python run_stage42_source_diversity_acquisition_package.py` passed, focused CD/CC/BV tests passed with 12 tests, and `.venv-pytorch/bin/python -m pytest tests` passed with 532 tests.

## Stage42-CE Source Diversity Conversion Preflight

```text
source = fresh_stage42_ce_source_diversity_conversion_preflight
verdict = stage42_ce_source_diversity_conversion_preflight_pass
gates = 12 / 12
targets_checked = 5
targets_with_local_path = 4
targets_with_schema_possible = 4
targets_with_t50_files = 3
targets_with_t100_files = 3
targets_with_independent_t50_candidates = 0
targets_source_cv_ready_now = 0
converted_datasets_now = 0
evaluated_datasets_now = 0
source_diversity_repair_ready_now = false
stage5c_executed = false
smc_enabled = false
```

Stage42-CE inspects the local paths from the CD acquisition package and performs a conversion preflight without converting or evaluating any dataset. UCY, ETH/BIWI, TrajNet/OpenTraj-related local paths are present and parseable, but none are source-CV ready: UCY/ETH/TrajNet still need official terms verification and source identity handling, OpenTraj is a toolkit/root scan rather than blanket data permission, and no target has an independent ready-to-claim t50 source. This makes the next source-diversity repair path more mechanical while preserving the blocker.

Artifacts:

- `outputs/stage42_long_research/source_diversity_conversion_preflight_stage42.md`
- `outputs/stage42_long_research/source_diversity_conversion_preflight_stage42.json`
- `outputs/stage42_long_research/stage42_stage_ce_gate.md`
- `outputs/stage42_long_research/user_action_required_source_conversion_preflight_stage42.md`

Verification: `.venv-pytorch/bin/python run_stage42_source_diversity_conversion_preflight.py` passed, focused CE/CD/CC tests passed with 14 tests, and `.venv-pytorch/bin/python -m pytest tests` passed with 536 tests.

## Stage42-CF Source Conversion Legal Gate

```text
source = fresh_stage42_cf_source_conversion_legal_gate
verdict = stage42_cf_source_conversion_legal_gate_pass
gates = 13 / 13
targets_checked = 5
local_paths_present = 4
schema_possible_targets = 4
targets_with_t50_files = 3
targets_with_t100_files = 3
source_cv_ready_now = 0
conversion_allowed_now_count = 0
converted_datasets_now = 0
evaluated_datasets_now = 0
stage5c_executed = false
smc_enabled = false
```

Stage42-CF adds a hard legal/source-identity gate before any source-diversity conversion. It reads the Stage42-CE local parseability evidence and deliberately allows zero conversions now: UCY/ETH/BIWI/TrajNet still require explicit official terms/path verification, OpenTraj is a toolkit/root scan rather than independent data permission, and the available t50-capable files do not yet form an independent source-CV-ready held-out source.

The generated `source_terms_confirmation_template_stage42.json` is a checklist, not permission. This keeps the next conversion path enforceable: explicit terms confirmation plus independent source identity must exist before any future conversion/no-leakage/source-CV/final-test stage can run.

Artifacts:

- `outputs/stage42_long_research/source_conversion_legal_gate_stage42.md`
- `outputs/stage42_long_research/source_conversion_legal_gate_stage42.json`
- `outputs/stage42_long_research/stage42_stage_cf_gate.md`
- `outputs/stage42_long_research/source_terms_confirmation_template_stage42.json`
- `outputs/stage42_long_research/user_action_required_source_legal_gate_stage42.md`

Verification: `.venv-pytorch/bin/python run_stage42_source_conversion_legal_gate.py` passed, focused CF/CE/CD/CC tests passed with 18 tests, and `.venv-pytorch/bin/python -m pytest tests` passed with 540 tests.

## Stage42-CG Source Terms Confirmation Validator

```text
source = fresh_stage42_cg_source_terms_confirmation_validator
verdict = stage42_cg_source_terms_confirmation_validator_pass
gates = 11 / 11
targets_validated = 5
terms_accepted_targets = 0
conversion_ready_targets = 0
conversion_allowed_now_count = 0
converted_datasets_now = 0
evaluated_datasets_now = 0
stage5c_executed = false
smc_enabled = false
```

Stage42-CG validates the CF-generated terms confirmation template and writes a conversion readiness manifest. Because the template is blank and no explicit official terms/path/source-identity confirmation has been supplied, all five targets remain blocked. This is the correct safety behavior: parseable local files and a template are not permission, and no future conversion should proceed until this validator reports ready targets and a later no-leakage/source-CV conversion gate passes.

Artifacts:

- `outputs/stage42_long_research/source_terms_validation_stage42.md`
- `outputs/stage42_long_research/source_terms_validation_stage42.json`
- `outputs/stage42_long_research/source_conversion_readiness_manifest_stage42.md`
- `outputs/stage42_long_research/source_conversion_readiness_manifest_stage42.json`
- `outputs/stage42_long_research/stage42_stage_cg_gate.md`
- `outputs/stage42_long_research/user_action_required_source_terms_validation_stage42.md`

Verification: `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py` passed, focused CG/CF/CE/CD/CC tests passed with 23 tests, and `.venv-pytorch/bin/python -m pytest tests` passed with 545 tests.

## Stage42-CH Metric/Time Claim Guard

```text
source = fresh_stage42_ch_metric_time_claim_guard
verdict = stage42_ch_metric_time_claim_guard_pass
gates = 11 / 11
datasets_audited = 7
source_records_audited = 7
source_specific_metric_time_candidates = 6
conversion_ready_targets = 0
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
restricted_subset_metric_seconds_claim_allowed_now = false
stage5c_executed = false
smc_enabled = false
```

Stage42-CH turns the existing source time/geometry calibration into a hard paper-claim guard. It identifies six ETH/UCY source-specific metric/time calibration candidates with 2.5fps / 0.4s annotation evidence, but because Stage42-CG has zero conversion-ready targets, no paper metric/seconds claim is currently allowed. This means any current M3W-Neural v1 paper package must keep the global claim as protected dataset-local/raw-frame 2.5D. SDD remains pixel raw-frame; TGSIM remains traffic diagnostic only.

Artifacts:

- `outputs/stage42_long_research/metric_time_claim_guard_stage42.md`
- `outputs/stage42_long_research/metric_time_claim_guard_stage42.json`
- `outputs/stage42_long_research/stage42_stage_ch_gate.md`
- `outputs/stage42_long_research/user_action_required_metric_time_claim_guard_stage42.md`

Verification: `.venv-pytorch/bin/python run_stage42_metric_time_claim_guard.py` passed, focused CH/CG/BN/A-calibration tests passed with 16 tests, and `.venv-pytorch/bin/python -m pytest tests` passed with 549 tests.

## Stage42-Z Post-CH Paper Claim Evidence Audit Refresh

```text
source = fresh_audit_from_stage42_wxy_and_paper_package_artifacts
verdict = stage42_z_paper_claim_evidence_audit_pass
gates = 22 / 22
claim_rows = 13
source_terms_validator_present = true
metric_time_guard_present = true
legal_conversion_not_overclaimed = true
restricted_metric_time_not_overclaimed = true
stage5c_executed = false
smc_enabled = false
```

Stage42-Z was refreshed after CG/CH so that the paper claim matrix now includes both legal conversion blockers and metric/time blockers. The audit keeps the positive protected 2.5D raw-frame claims, but explicitly rejects two tempting overclaims: source-diversity conversion cannot be counted as converted/evaluated data (`terms_accepted=0`, `conversion_ready=0`, `converted=0`, `evaluated=0`), and restricted ETH/UCY metric/seconds subset claims are still blocked despite six source-specific calibration candidates.

Artifacts:

- `outputs/stage42_long_research/paper_claim_evidence_audit_stage42.md`
- `outputs/stage42_long_research/paper_claim_evidence_audit_stage42.json`
- `outputs/stage42_long_research/paper_claim_evidence_audit_stage42.csv`
- `outputs/stage42_long_research/stage42_stage_z_gate.md`

Verification: `.venv-pytorch/bin/python run_stage42_paper_claim_evidence_audit.py` passed and focused Z/CH/CG tests passed with 12 tests.

## Stage42-CI Context Contribution Forensics

```text
source = fresh_synthesis_from_stage42_ablation_and_claim_audits
verdict = stage42_ci_context_contribution_forensics_pass
gates = 13 / 13
dominant mechanism = baseline_family_rollout_context
supported core component = history_tokens
supported secondary component = domain_expert
mixed/not-main = goal_scene_context, neighbor_interaction_context
not independent main claim = JEPA, Transformer
Stage5C = false
SMC = false
```

Stage42-CI turns the mixed context evidence into a stricter contribution map. It confirms that the current working mechanism is baseline-family rollout context under Stage37/teacher floor, with causal sequence history as the strongest positive context component and domain expert as a smaller guarded component. It also prevents overclaiming: goal/scene has standalone signal but no reliable incremental gain after baseline-family context, neighbor/interaction is weak under current graph/sequence protocols, JEPA remains diagnostic, and Transformer is not yet an independent deployable world-dynamics claim.

Artifacts:

- `outputs/stage42_long_research/context_contribution_forensics_stage42.md`
- `outputs/stage42_long_research/context_contribution_forensics_stage42.json`
- `outputs/stage42_long_research/stage42_stage_ci_gate.md`

Verification: `.venv-pytorch/bin/python run_stage42_context_contribution_forensics.py` passed and focused CI/Z tests passed with 6 tests.

<!-- STAGE42_CS_FROZEN_PROXIMITY_GUARD_POLICY:START -->
## Stage42-CS Frozen Proximity-Guard Composer Policy

- source: `fresh_policy_freeze_from_stage42_cq_cr`
- verdict: `stage42_cs_frozen_proximity_guard_policy_pass`
- gates: `25 / 25`
- policy artifact: `outputs/stage42_long_research/frozen_proximity_guard_composer_policy_stage42_policy.json`
- policy hash: `4af6536f86499d5b39efa535bb81978398586d65746bb983571b642af7c92d59`
- selected deployment role: `safety_sensitive_deployable_composer_variant`
- ADE vs endpoint-linear all/t50/t100 raw/hard: `1.77%` / `1.07%` / `3.48%` / `1.93%`
- easy degradation: `0.25%`
- near-collision@0.05 delta vs endpoint-linear: `-0.06%`
- This freezes the Stage42-CQ/CR safety-sensitive composer. The no-guard composer remains accuracy-priority diagnostic only.
- Claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation claim, no global metric/seconds-level claim, no Stage5C execution, no SMC.
<!-- STAGE42_CS_FROZEN_PROXIMITY_GUARD_POLICY:END -->

<!-- STAGE42_CT_FROZEN_POLICY_REPLAY:START -->
## Stage42-CT Frozen Policy Replay / Reproducibility Verifier

- source: `fresh_replay_from_frozen_policy_artifact`
- verdict: `stage42_ct_frozen_policy_replay_pass`
- gates: `30 / 30`
- replayed policy artifact: `outputs/stage42_long_research/frozen_proximity_guard_composer_policy_stage42_policy.json`
- policy hash: `4af6536f86499d5b39efa535bb81978398586d65746bb983571b642af7c92d59`
- replay check: policy artifact matches Stage42-CS embedded policy and Stage42-CQ source metrics/safety.
- ADE vs endpoint-linear all/t50/t100 raw/hard: `1.77%` / `1.07%` / `3.48%` / `1.93%`
- easy degradation: `0.25%`
- near-collision@0.05 delta vs endpoint-linear: `-0.06%`
- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_CT_FROZEN_POLICY_REPLAY:END -->

<!-- STAGE42_CU_RUNTIME_POLICY_API:START -->
## Stage42-CU Runtime Policy API Smoke Audit

- source: `fresh_runtime_api_from_frozen_policy_artifact`
- verdict: `stage42_cu_runtime_policy_api_pass`
- gates: `19 / 19`
- policy hash: `4af6536f86499d5b39efa535bb81978398586d65746bb983571b642af7c92d59`
- runtime inputs: domain, horizon, endpoint predicted group min-distance, full-waypoint predicted group min-distance.
- guard rule: use full-waypoint only when validation-selected base slice wants full and predicted proximity guard does not fire.
- smoke cases: guard-clear full slice, guarded-off full slice, endpoint-only slice, and nonfinite-geometry replay behavior all pass.
- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_CU_RUNTIME_POLICY_API:END -->

<!-- STAGE42_CV_BATCH_RUNTIME_REPLAY:START -->
## Stage42-CV Batch Runtime Policy Replay

- source: `fresh_batch_runtime_replay_from_frozen_policy_artifact`
- verdict: `stage42_cv_batch_runtime_replay_pass`
- gates: `25 / 25`
- policy hash: `4af6536f86499d5b39efa535bb81978398586d65746bb983571b642af7c92d59`
- replay scope: real common validation/test rows, not toy smoke cases.
- replay result: validation and test runtime decisions exactly match the original CQ guard output.
- test ADE vs endpoint-linear all/t50/t100 raw/hard: `1.77%` / `1.07%` / `3.48%` / `1.93%`
- test easy degradation: `0.25%`
- near-collision@0.05 delta vs endpoint-linear: `-0.06%`
- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_CV_BATCH_RUNTIME_REPLAY:END -->

<!-- STAGE42_CW_RUNTIME_REPLAY_PAPER_REFRESH:START -->
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
<!-- STAGE42_CW_RUNTIME_REPLAY_PAPER_REFRESH:END -->

<!-- STAGE42_CX_EVIDENCE_PROVENANCE:START -->
## Stage42-CX Evidence Provenance / Command Matrix

- source: `fresh_evidence_provenance_from_stage42_artifacts`
- role: paper-ready provenance and reproducibility audit.
- gate: `20 / 20`; verdict `stage42_cx_evidence_provenance_pass`.
- artifacts audited: `25`.
- artifacts with passing gates: `25`.
- source-label counts: `{'fresh_run': 24, 'cached_verified': 1}`.
- worktree caveat artifacts recorded: `4`.
- Dirty/untracked generated files are not hidden; they are recorded as caveats and must not be treated as extra clean paper evidence.
- This audit does not create metric/seconds/3D/foundation claims and does not execute Stage5C or SMC.
<!-- STAGE42_CX_EVIDENCE_PROVENANCE:END -->

<!-- STAGE42_CY_WORKTREE_CAVEAT_CLASSIFIER:START -->
## Stage42-CY Worktree Caveat Classifier

- source: `fresh_worktree_caveat_classification`
- role: classify dirty tracked files before paper-freeze evidence claims.
- gate: `11 / 11`; verdict `stage42_cy_worktree_caveat_classifier_pass`.
- tracked dirty files inspected: `8`.
- Stage42 dirty files inspected: `0`.
- Stage42 substantive dirty files: `0`.
- allowed classifications: `{'substantive_json_change': 8}`.
- Metadata-only, paper-size-only, and append-only ledger changes are recorded as caveats, not new model evidence.
- This classifier does not execute Stage5C, does not enable SMC, and does not create metric/seconds/3D/foundation claims.
<!-- STAGE42_CY_WORKTREE_CAVEAT_CLASSIFIER:END -->

<!-- STAGE42_CZ_PAPER_FREEZE_MANIFEST:START -->
## Stage42-CZ Paper Freeze Candidate Manifest

- source: `fresh_freeze_candidate_manifest_from_cx_cy`
- role: hash manifest for the current Stage42 paper evidence candidate.
- gate: `15 / 15`; verdict `stage42_cz_paper_freeze_candidate_manifest_pass`.
- freeze status: `candidate_clean`.
- final immutable release: `True`.
- files hashed: `87`.
- metadata caveats: `0`; substantive caveats: `0`.
- This is a paper evidence freeze candidate under protected dataset-local/raw-frame 2.5D boundaries.
- It is not true 3D, not foundation, not metric/seconds-level, not Stage5C, and not SMC.
<!-- STAGE42_CZ_PAPER_FREEZE_MANIFEST:END -->

<!-- STAGE42_TEST_ISOLATION_ARTIFACT_HYGIENE:START -->
## Stage42 Report-Test Artifact Hygiene

- commit: `08a8b2a Isolate Stage42 report tests from tracked artifacts`.
- role: reproducibility hygiene for the Stage42 paper/evidence package.
- change: Stage42 report-writing pytest cases now monkeypatch outputs into `tmp_path` instead of rewriting tracked `outputs/stage42_long_research/*` paper artifacts and `run_ledger.jsonl`.
- focused verification: `13 passed`.
- full verification: `615 passed`.
- post-fix worktree caveat classifier: Stage42 dirty tracked files `0`; Stage42 substantive dirty files `0`; remaining tracked dirty files are historical Stage17-19 outside-scope report drift.
- This is not a new model result and does not alter the supported claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_TEST_ISOLATION_ARTIFACT_HYGIENE:END -->

<!-- STAGE42_DA_NEXT_ACTION_QUEUE:START -->
## Stage42-DA Next-Action Evidence Queue

- source: `fresh_synthesis_from_cached_verified_stage42_artifacts`
- role: convert current Stage42 paper gaps into prioritized executable next actions.
- gate: `15 / 15`; verdict `stage42_da_next_action_queue_pass`.
- top priority: `DA-1 Close legal/source support for ETH_UCY and TrajNet t100/t50 calibration`.
- user/external blockers remain explicit; no not_run item is counted complete.
- Current deployable claim remains protected dataset-local/raw-frame 2.5D; no true 3D, no foundation, no metric/seconds-level, no Stage5C, no SMC.
<!-- STAGE42_DA_NEXT_ACTION_QUEUE:END -->

<!-- STAGE42_DB_CONTEXT_RESCUE_DECISION:START -->
## Stage42-DB Context Rescue Decision Audit

- source: `fresh_synthesis_from_cached_verified_context_runs`
- role: decide whether existing goal/scene, neighbor/interaction, sequence, and graph context protocols should be repeated.
- gate: `13 / 13`; verdict `stage42_db_context_rescue_decision_pass`.
- decision: `stop_repeating_current_context_residual_or_gated_protocols`.
- best delta all/t50/hard vs baseline-family control: `-0.0230` / `-0.0831` / `-0.0262`.
- No safe positive context variant was found under the existing residual/gated protocols; next work must change target/model/data, not just rerun thresholds.
- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; no true 3D, foundation, metric/seconds-level, Stage5C, or SMC.
<!-- STAGE42_DB_CONTEXT_RESCUE_DECISION:END -->

<!-- STAGE42_DC_CONTEXT_SWITCHABILITY_GATE:START -->
## Stage42-DC Context Switchability / Gain-Harm Gate

- source: `fresh_run`
- role: change context supervision from waypoint residual to gain/harm switchability after Stage42-DB no-go.
- gate: `15 / 15`; verdict `stage42_dc_context_switchability_gate_pass`.
- selected candidate: `baseline_plus_knn_graph`; decision `context_switchability_not_supported`.
- delta vs baseline-family all/t50/hard/easy: `0.0004` / `-0.0001` / `0.0004` / `-0.0024`.
- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; no true 3D, foundation, metric/seconds-level, Stage5C, or SMC.
<!-- STAGE42_DC_CONTEXT_SWITCHABILITY_GATE:END -->

<!-- STAGE42_DD_SOURCE_SUPPORT_CLOSURE_AUDIT:START -->
## Stage42-DD Source Support Closure Audit

- source: `fresh_stage42_dd_source_support_closure_audit`
- role: close or explicitly block DA-1 legal/source/time-calibration support for ETH_UCY, TrajNet, and UCY.
- gate: `15 / 15`; verdict `stage42_dd_source_support_closure_audit_pass_open_blockers`.
- domains_not_closed: `['ETH_UCY', 'TrajNet', 'UCY']`.
- restricted ETH/UCY source-specific metric/time candidates exist, but global metric/seconds and global t100 deployable claims remain blocked.
- User/external action remains required before official converted/evaluated metric-time or t100 source-CV claims.
- Stage5C remains false; SMC remains false.
<!-- STAGE42_DD_SOURCE_SUPPORT_CLOSURE_AUDIT:END -->

<!-- STAGE42_DE_FULL_WAYPOINT_DEPLOYMENT_GAP_AUDIT:START -->
## Stage42-DE Full-Waypoint Deployment Gap Audit

- source: `fresh_stage42_de_full_waypoint_deployment_gap_audit`
- role: decide whether full-waypoint can be promoted from auxiliary/composer evidence to primary deployable world dynamics.
- gate: `17 / 17`; verdict `stage42_de_full_waypoint_deployment_gap_audit_pass_primary_promotion_blocked`.
- decision: `protected_full_waypoint_composer_supported_deployment_promotion_blocked`.
- horizon_auxiliary_supported: `True`; guarded_composer_supported: `True`.
- primary deployable full-waypoint promotion: `False`.
- blockers: `['protected_full_waypoint_does_not_beat_endpoint_linear_on_all_and_hard', 'ungated_full_waypoint_easy_degradation_unsafe', 'source_legal_time_t100_closure_open', 'graph_group_interaction_has_proximity_caveat']`.
- Conclusion: keep Stage37/teacher or endpoint-linear safety floor; use guarded full-waypoint composer only as protected horizon/shape component until all/hard/proximity/source-support gaps are closed.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DE_FULL_WAYPOINT_DEPLOYMENT_GAP_AUDIT:END -->

<!-- STAGE42_DF_FULL_WAYPOINT_ALL_HARD_PROXIMITY_REPAIR:START -->
## Stage42-DF All-Hard / Proximity Full-Waypoint Repair

- source: `fresh_stage42_df_all_hard_proximity_full_waypoint_repair`
- role: validation-only repair search for the Stage42-DE all/hard/proximity full-waypoint deployment blocker.
- gate: `12 / 14`; verdict `stage42_df_all_hard_proximity_repair_partial`.
- test vs endpoint-linear: all `-0.67%`, t50 `-1.40%`, t100 raw `-0.66%`, hard `-0.72%`, easy `0.19%`.
- delta vs Stage42-CQ: all `-2.44%`, t50 `-2.46%`, t100 raw `-4.14%`, hard `-2.65%`, near@0.05 `-0.05%`.
- decision: `all_hard_proximity_repair_no_primary_promotion_keep_cq_guarded_composer`.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DF_FULL_WAYPOINT_ALL_HARD_PROXIMITY_REPAIR:END -->

<!-- STAGE42_DG_FULL_WAYPOINT_ALL_HARD_LOSS_REPAIR:START -->
## Stage42-DG Full-Waypoint All/Hard Weighted Loss Repair

- source: `fresh_stage42_dg_full_waypoint_all_hard_loss_repair`
- role: actual retraining probe for all/hard/long-horizon weighted full-waypoint dynamics, following Stage42-DE/DF blockers.
- selected loss variant: `balanced` with lambda `100.0`.
- gate: `13 / 15`; verdict `stage42_dg_full_waypoint_weighted_loss_repair_pass_positive_not_better_than_am`.
- test vs train-horizon causal floor: all `24.58%`, t50 `22.02%`, t100 raw `14.37%`, hard `23.75%`, easy `-25.66%`.
- delta vs Stage42-AM: all `0.00%`, t50 `0.00%`, t100 raw `0.00%`, hard `0.00%`, easy `0.00%`.
- decision: `weighted_loss_not_enough_keep_stage42_am_or_cq_floor`.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DG_FULL_WAYPOINT_ALL_HARD_LOSS_REPAIR:END -->

<!-- STAGE42_DH_FULL_WAYPOINT_PROXIMITY_OCCUPANCY_LOSS_REPAIR:START -->
## Stage42-DH Full-Waypoint Proximity / Occupancy-Proxy Loss Repair

- source: `fresh_stage42_dh_full_waypoint_proximity_occupancy_loss_repair`
- role: actual retraining probe for proximity/density/occupancy-proxy weighted full-waypoint dynamics after Stage42-DE/DF/DG blockers.
- selected candidate: `proximity_close_weighted` with `stage42_am_features` and lambda `100.0`.
- gate: `15 / 16`; verdict `stage42_dh_proximity_occupancy_loss_repair_pass_positive_not_better_than_am`.
- test vs train-horizon causal floor: all `25.51%`, t50 `22.14%`, t100 raw `14.34%`, hard `23.74%`, easy `-29.23%`.
- delta vs Stage42-AM: all `0.93%`, t50 `0.12%`, t100 raw `-0.03%`, hard `-0.01%`, easy `-3.57%`.
- decision: `proximity_occupancy_loss_not_enough_keep_stage42_am_or_cq_floor`.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DH_FULL_WAYPOINT_PROXIMITY_OCCUPANCY_LOSS_REPAIR:END -->

<!-- STAGE42_DI_GROUP_CONSISTENCY_FULL_WAYPOINT_REPAIR:START -->
## Stage42-DI Group-Consistency Full-Waypoint Repair

- source: `fresh_stage42_di_group_consistency_full_waypoint_repair`
- role: explicit all-agent group-consistency / proximity repair over source-level full-waypoint predictions after Stage42-DE/DF/DG/DH blockers.
- selected repair: `{'mode': 'repel_unsafe', 'min_sep': 0.08, 'margin': 0.0, 'strength': 0.5}`.
- gate: `17 / 17`; verdict `stage42_di_group_consistency_full_waypoint_repair_pass_promotable`.
- test vs train-horizon causal floor: all `24.72%`, t50 `22.36%`, t100 raw `14.35%`, hard `23.89%`, easy `-25.63%`.
- delta vs Stage42-AM: all `0.14%`, t50 `0.35%`, t100 raw `-0.02%`, hard `0.14%`, easy `0.03%`.
- near@0.05 base/final/floor: `1.94%` / `1.38%` / `2.24%`.
- decision: `promote_stage42_di_group_consistency_full_waypoint_repair`.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DI_GROUP_CONSISTENCY_FULL_WAYPOINT_REPAIR:END -->

<!-- STAGE42_DJ_FROZEN_GROUP_CONSISTENCY_POLICY:START -->
## Stage42-DJ Frozen Group-Consistency Full-Waypoint Policy

- source: `fresh_policy_freeze_from_stage42_di`
- role: freeze the Stage42-DI promoted group-consistency full-waypoint repair as a reproducible deployment/paper artifact.
- policy artifact: `outputs/stage42_long_research/frozen_group_consistency_full_waypoint_policy_stage42_policy.json`
- policy hash: `617ef9952b1439f3678318129a4979c7a171f2ba882742cd18acd46c5ae92141`
- gate: `22 / 22`; verdict `stage42_dj_frozen_group_consistency_policy_pass`.
- test vs train-horizon causal floor: all `24.72%`, t50 `22.36%`, t100 raw `14.35%`, hard `23.89%`, easy `-25.63%`.
- delta vs Stage42-AM all/t50/hard: `0.14%` / `0.35%` / `0.14%`.
- near@0.05 base/final/floor: `1.94%` / `1.38%` / `2.24%`.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DJ_FROZEN_GROUP_CONSISTENCY_POLICY:END -->

<!-- STAGE42_DK_GROUP_CONSISTENCY_POLICY_REPLAY:START -->
## Stage42-DK Group-Consistency Policy Replay

- source: `fresh_replay_from_frozen_group_consistency_policy_artifact`
- verdict: `stage42_dk_group_consistency_policy_replay_pass`
- gates: `34 / 34`
- replayed policy artifact: `outputs/stage42_long_research/frozen_group_consistency_full_waypoint_policy_stage42_policy.json`
- policy hash: `617ef9952b1439f3678318129a4979c7a171f2ba882742cd18acd46c5ae92141`
- replay check: policy artifact matches Stage42-DJ embedded policy and Stage42-DI selected repair / metrics / safety.
- ADE vs train-horizon causal floor all/t50/t100 raw/hard: `24.72%` / `22.36%` / `14.35%` / `23.89%`
- easy degradation: `-25.63%`
- near@0.05 base/final/floor: `1.94%` / `1.38%` / `2.24%`
- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
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

<!-- STAGE42_DM_REVIEWER_REPLAY_PACKAGE:START -->
## Stage42-DM Reviewer Replay Package

- source: `fresh_reviewer_replay_package_from_stage42_runtime_and_manifest_artifacts`
- role: reviewer-facing minimal replay package for provenance, manifest, and runtime policy exact replay.
- gate: `21 / 21`; verdict `stage42_dm_reviewer_replay_package_pass`.
- commands file: `outputs/stage42_long_research/reviewer_replay_commands_stage42.sh`.
- group-consistency runtime all/t50/t100 raw/hard: `0.24715658317833844` / `0.2236298792899738` / `0.1434611214781808` / `0.23887420070464105`.
- This is replay/provenance packaging only: no training, no threshold tuning, no Stage5C, no SMC, no metric/seconds-level claim.
<!-- STAGE42_DM_REVIEWER_REPLAY_PACKAGE:END -->

<!-- STAGE42_DN_DEPLOYMENT_VARIANT_CARD:START -->
## Stage42-DN Deployment Variant Card

- source: `fresh_deployment_variant_card_from_stage42_cr_cq_di_dl_dm`
- role: separates safety-sensitive deployment, accuracy-priority diagnostics, and protocol-specific group-consistency runtime policy.
- gate: `20 / 20`; verdict `stage42_dn_deployment_variant_card_pass`.
- safety-sensitive default: `proximity_guard` for endpoint-linear bridge/shape deployment with joint-proximity safety.
- strongest full-waypoint runtime evidence: `group_consistency_full_waypoint_runtime`, but it uses train-horizon causal-floor comparison and must not be rank-mixed with endpoint-linear composer variants without that caveat.
- accuracy-priority diagnostic: `no_proximity_guard`; it has higher ADE gains but worsens near-collision@0.05 and is not the safety-sensitive deployment claim.
- No Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.
<!-- STAGE42_DN_DEPLOYMENT_VARIANT_CARD:END -->

<!-- STAGE42_DO_SOURCE_LEGAL_TIME_ACTION_PACKAGE:START -->
## Stage42-DO Source Legal/Time Action Package

- source: `fresh_synthesis_from_stage42_cg_bn_dd_after_da1_rerun`
- role: closes the current DA-1 pass as an honest blocker/action package, not as conversion or evaluation.
- gate: `13 / 13`; verdict `stage42_do_source_legal_time_action_package_pass`.
- conversion-ready targets: `0`; converted/evaluated now: `0` / `0`.
- source-specific metric/time candidate count: `6`.
- global metric/seconds/t100 deployable claims remain blocked; Stage5C and SMC remain disabled.
- user action file: `outputs/stage42_long_research/user_action_required_source_legal_time_stage42.md`.
<!-- STAGE42_DO_SOURCE_LEGAL_TIME_ACTION_PACKAGE:END -->

<!-- STAGE42_DP_CONTEXT_MODEL_CLOSURE:START -->
## Stage42-DP Context Model Closure

- source: `fresh_synthesis_after_fresh_ar_as_rerun`
- verdict: `stage42_dp_context_model_closure_pass`; gates `19 / 19`.
- fresh reruns: Stage42-AR sequence context and Stage42-AS graph context.
- closure decision: `close_current_sequence_graph_residual_context_protocol`.
- best delta all/t50/hard vs baseline-family control: `-0.0230` / `-0.0831` / `-0.0262`.
- conclusion: current residual sequence/graph context protocol does not add independent lift beyond baseline-family rollout context.
- next: change target/data/model before revisiting context, and keep protected Stage37/teacher/runtime policies as deployable floor.
- Claim boundary: dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no global metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_DP_CONTEXT_MODEL_CLOSURE:END -->

<!-- STAGE42_DQ_FULL_WAYPOINT_PROMOTION_CHECKPOINT:START -->
## Stage42-DQ Full-Waypoint Promotion Checkpoint

- source: `fresh_synthesis_after_da3_full_waypoint_rerun`
- verdict: `stage42_dq_full_waypoint_promotion_checkpoint_pass`; gates `24 / 24`.
- fresh chain: Stage42-C full-waypoint dynamics, Stage42-CO common-validation composer, Stage42-DI group-consistency repair, Stage42-DL runtime replay.
- group-consistency runtime vs train-horizon causal floor all/t50/t100 raw/hard: `24.72%` / `22.36%` / `14.35%` / `23.89%`.
- runtime replay exact: switch `True`, selected_xy max abs diff `0.0`.
- near@0.05 base/final/floor: `1.94%` / `1.38%` / `2.24%`.
- promotion: protected source-level group-consistency full-waypoint runtime policy is supported; ungated full-waypoint and global primary replacement remain blocked.
- Claim boundary: dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no global metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_DQ_FULL_WAYPOINT_PROMOTION_CHECKPOINT:END -->

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

<!-- STAGE42_DS_SOURCE_CONVERSION_READINESS_RECHECK:START -->
## Stage42-DS Source Conversion Readiness Recheck

- source: `fresh_local_path_scan_after_stage42_do`
- role: separates local raw-path/derived-cache hints from legal conversion readiness.
- gate: `13 / 13`; verdict `stage42_ds_source_conversion_readiness_recheck_pass`.
- targets checked: `7`; raw-path found: `6`; derived-cache found: `6`.
- technical preflight possible: `6`; conversion-ready targets: `0`.
- No dataset was converted or evaluated in this step; legal/source blockers remain preserved.
- report: `outputs/stage42_long_research/source_conversion_readiness_recheck_stage42.md`.
<!-- STAGE42_DS_SOURCE_CONVERSION_READINESS_RECHECK:END -->

<!-- STAGE42_DT_RAW_SOURCE_PARSEABILITY_DRY_RUN:START -->
## Stage42-DT Raw Source Parseability Dry Run

- source: `fresh_sample_only_raw_source_parseability_dry_run`
- role: sample-only technical parser preflight after Stage42-DS; no conversion, no evaluation.
- gate: `11 / 11`; verdict `stage42_dt_raw_source_parseability_dry_run_pass`.
- dry-run parseable targets: `4`; targets with homography/time hints: `2`.
- legal conversion ready targets: `0`; generated rows: `0`.
- Homography/time hints remain hints only; no metric/seconds claim is made.
- report: `outputs/stage42_long_research/raw_source_parseability_dry_run_stage42.md`.
<!-- STAGE42_DT_RAW_SOURCE_PARSEABILITY_DRY_RUN:END -->

<!-- STAGE42_DU_RAW_SOURCE_TIME_GEOMETRY_HINT_AUDIT:START -->
## Stage42-DU Raw Source Time/Geometry Hint Audit

- source: `fresh_hint_audit_from_local_raw_sources_after_stage42_dt`
- role: extracts H/FPS/stride hints only; no conversion, no evaluation, no metric/seconds claim.
- gate: `14 / 14`; verdict `stage42_du_raw_source_time_geometry_hint_audit_pass`.
- H-hint targets: `2`; time-hint targets: `3`; stride-hint targets: `4`.
- metric/time subset hint targets: `2`; legal conversion ready targets: `0`.
- report: `outputs/stage42_long_research/raw_source_time_geometry_hint_audit_stage42.md`.
<!-- STAGE42_DU_RAW_SOURCE_TIME_GEOMETRY_HINT_AUDIT:END -->

<!-- STAGE42_DV_CALIBRATION_CANDIDATE_MANIFEST:START -->
## Stage42-DV Calibration Candidate Manifest

- source: `fresh_synthesis_from_stage42_du_bn`
- role: ranks source-specific calibration candidates from raw H/FPS/stride hints; no conversion/evaluation.
- gate: `13 / 13`; verdict `stage42_dv_calibration_candidate_manifest_pass`.
- source-specific candidate targets: `2`; time/stride candidate targets: `1`.
- conversion-ready targets: `0`; global metric/seconds claim remains `False`.
- report: `outputs/stage42_long_research/calibration_candidate_manifest_stage42.md`.
<!-- STAGE42_DV_CALIBRATION_CANDIDATE_MANIFEST:END -->

<!-- STAGE42_DW_SOURCE_SPECIFIC_CONVERSION_DRY_RUN:START -->
## Stage42-DW Source-Specific Conversion Dry-Run

- source: `fresh_source_specific_conversion_dry_run_from_stage42_dv`
- role: parses calibrated UCY/ETH source candidates for horizon/source-CV readiness; no conversion/evaluation.
- gate: `15 / 15`; verdict `stage42_dw_source_specific_conversion_dry_run_pass`.
- sources checked: `6`; technical ready after terms: `5`.
- technical not-ready sources: `['UCY_zara03']`.
- estimated t50/t100 windows: `10060` / `5696`.
- source-CV domains after terms: `['UCY']`; conversion allowed now remains `0`.
- report: `outputs/stage42_long_research/source_specific_conversion_dry_run_stage42.md`.
<!-- STAGE42_DW_SOURCE_SPECIFIC_CONVERSION_DRY_RUN:END -->

<!-- STAGE42_DX_FULL_WAYPOINT_LOSS_FAMILY_REPLAY:START -->
## Stage42-DX Full-Waypoint Loss-Family Fresh Replay

- source: `fresh_rerun_dg_dh_loss_family_replay`
- role: reruns DG/DH full-waypoint loss-family probes and applies one promotion gate over Stage42-AM.
- gate: `10 / 10`; verdict `stage42_dx_loss_family_replay_pass_blocker_confirmed`.
- best replay candidate: `proximity_occupancy_loss`; all `0.255061`, t50 `0.221366`, hard `0.237393`, easy `-0.292293`.
- promotion decision: `do_not_promote_keep_stage42_am_or_cq_floor`; blockers: `['no_loss_family_candidate_beats_stage42_am_on_all_and_hard', 'primary_full_waypoint_promotion_blocked', 'next_step_requires_model_architecture_or_explicit_physical_consistency_target_not_more_scalar_weighting']`.
- Stage5C remains false; SMC remains false; no metric/seconds claim.
<!-- STAGE42_DX_FULL_WAYPOINT_LOSS_FAMILY_REPLAY:END -->

<!-- STAGE42_DY_EXPLICIT_PHYSICAL_CONSISTENCY_CHECKPOINT:START -->
## Stage42-DY Explicit Physical Consistency Checkpoint

- source: `fresh_dg_dh_di_physical_consistency_checkpoint`
- role: follows Stage42-DX by comparing scalar loss-family replay with explicit group/physical consistency repair.
- gate: `16 / 16`; verdict `stage42_dy_explicit_physical_consistency_checkpoint_pass_source_level_promoted`.
- loss-family any promotable over Stage42-AM: `False`; best scalar candidate `proximity_occupancy_loss` all/t50/hard `0.255061` / `0.221366` / `0.237393`.
- group-consistency source-level policy all/t50/t100 raw/hard/easy `0.247157` / `0.223630` / `0.143461` / `0.238874` / `-0.256309`.
- group-consistency beats Stage42-AM on all/hard by `0.001368` / `0.001380` and repairs near@0.05 from `0.019364` to `0.013823`.
- deployment boundary: promote explicit group-consistency as source-level full-waypoint physical policy; do not claim global primary full-waypoint replacement, metric/seconds-level, Stage5C, or SMC.
<!-- STAGE42_DY_EXPLICIT_PHYSICAL_CONSISTENCY_CHECKPOINT:END -->

<!-- STAGE42_DZ_UCY_SUPPORTED_GROUP_CONSISTENCY:START -->
## Stage42-DZ UCY-Supported Group-Consistency Full-Waypoint Repair

- source: `fresh_ucy_internal_validation_group_consistency_repair`
- role: reruns explicit group/physical consistency on the UCY validation-supported split, addressing the prior TrajNet-only/floor-only domain boundary.
- gate: `15 / 15`; verdict `stage42_dz_ucy_supported_group_consistency_pass_dual_domain`.
- global all/t50/t100 raw/hard/easy `0.328904` / `0.269864` / `0.211165` / `0.318864` / `-0.320940`.
- positive safe domains: `2`; UCY all/t50/hard `0.355808` / `0.227206` / `0.337848`; TrajNet all/t50/hard `0.320715` / `0.281804` / `0.312868`.
- near@0.05 base/final `0.020797` / `0.013148`; still raw-frame/dataset-local, no metric/seconds claim, Stage5C false, SMC false.
<!-- STAGE42_DZ_UCY_SUPPORTED_GROUP_CONSISTENCY:END -->

<!-- STAGE42_EA_DUAL_DOMAIN_GROUP_CONSISTENCY_STATISTICS:START -->
## Stage42-EA Dual-Domain Group-Consistency Statistical Evidence

- source: `fresh_stage42_ea_dual_domain_group_consistency_statistics`
- role: fresh row-level 2000-bootstrap evidence for the Stage42-DZ UCY-supported group-consistency policy.
- gate: `12 / 12`; verdict `stage42_ea_dual_domain_group_consistency_statistics_pass`.
- global all/t50/hard CI lows: `0.325616` / `0.265328` / `0.315115`; easy high `-0.312813`.
- UCY all/t50/hard CI lows: `0.346983` / `0.213784` / `0.328373`; TrajNet all/t50/hard CI lows `0.317175` / `0.277244` / `0.308982`.
- near@0.05 final-base delta high `-0.006722`; raw-frame/dataset-local only; Stage5C false; SMC false.
<!-- STAGE42_EA_DUAL_DOMAIN_GROUP_CONSISTENCY_STATISTICS:END -->

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

<!-- STAGE42_EC_GROUP_CONSISTENCY_CONTRIBUTION_AUDIT:START -->
## Stage42-EC Group-Consistency Contribution Audit

- source: `fresh_synthesis_from_stage42_dy_dz_ea_dp`
- role: converts the latest positive and negative evidence into a contribution/claim matrix.
- gate: `17 / 17`; verdict `stage42_ec_group_consistency_contribution_audit_pass`.
- supported contribution: explicit group-consistency full-waypoint source-level repair, all/t50/t100 raw/hard `0.247157` / `0.223630` / `0.143461` / `0.238874`.
- dual-domain evidence: UCY all/t50/hard `0.355808` / `0.227206` / `0.337848`; TrajNet all/t50/hard `0.320715` / `0.281804` / `0.312868`.
- bootstrap CI lows global all/t50/hard `0.325616` / `0.265328` / `0.315115`; easy high `-0.312813`.
- blocked contributions: scalar loss-family primary `blocked`, current sequence/graph residual context `closed_current_protocol`, goal/scene main claim `not_supported_under_current_protocols`, neighbor/interaction main claim `not_supported_under_current_protocols`.
- claim boundary: supported as protected source-level raw-frame full-waypoint evidence only; no true-3D, foundation, metric/seconds, Stage5C, SMC, or ungated/global primary replacement claim.
<!-- STAGE42_EC_GROUP_CONSISTENCY_CONTRIBUTION_AUDIT:END -->

<!-- STAGE42_ED_SOURCE_CONVERSION_UNBLOCKER:START -->
## Stage42-ED Source Conversion Unblocker Package

- source: `fresh_synthesis_from_stage42_cg_dw_do_ds`
- role: convert local parseability/source-specific calibration hints into exact user actions; no download/conversion/evaluation.
- gate: `15 / 15`; verdict `stage42_ed_source_conversion_unblocker_pass`.
- conversion_ready_now: `0`; conversion_allowed_now: `0`; converted/evaluated now `0` / `0`.
- technical_ready_after_terms_targets: `2`; estimated t50/t100 windows after terms `10060` / `5696`.
- domains_with_source_cv_after_terms: `['UCY']`; first unblock targets remain UCY and ETH/BIWI terms/path/source identity.
- boundary: local path and parseability are not legal conversion; metric/seconds, Stage5C, and SMC remain blocked.
<!-- STAGE42_ED_SOURCE_CONVERSION_UNBLOCKER:END -->

<!-- STAGE42_EE_CONTEXT_SWITCHABILITY_MATERIALITY:START -->
## Stage42-EE Context Switchability Materiality Audit

- source: `fresh_rerun_stage42_dc_context_switchability_materiality`
- role: fresh-reruns gain/harm context switchability and applies a 1pp materiality threshold.
- gate: `12 / 12`; verdict `stage42_ee_context_switchability_materiality_audit_pass`.
- selected context candidate `baseline_plus_knn_graph` delta all/t50/hard/easy `0.000368` / `-0.000074` / `0.000424` / `-0.002388`.
- material_context_contribution: `False`; decision `context_switchability_materiality_blocked`.
- boundary: current context switchability has micro-deltas only, so scene/goal/neighbor/interaction main claims remain blocked under this protocol.
<!-- STAGE42_EE_CONTEXT_SWITCHABILITY_MATERIALITY:END -->

<!-- STAGE42_EF_SOURCE_TERMS_GAP_AUDIT:START -->
## Stage42-EF Source Terms Gap Audit

- source: `fresh_rerun_cg_plus_ed_source_terms_gap_audit`
- role: reruns source terms validator and merges it with ED technical-after-terms potential.
- gate: `13 / 13`; verdict `stage42_ef_source_terms_gap_audit_pass`.
- conversion_ready_now: `0`; converted/evaluated now `0` / `0`.
- top unblock targets: `['ucy_crowd_original', 'eth_biwi_original', 'aerialmpt_or_other_topdown']`; estimated t50/t100 after terms `10060` / `5696`.
- boundary: no legal conversion, no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_EF_SOURCE_TERMS_GAP_AUDIT:END -->

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

<!-- STAGE42_EH_SOURCE_TERMS_CONFIRMATION_INTAKE:START -->
## Stage42-EH Source Terms Confirmation Intake Package

- source: `fresh_source_terms_confirmation_intake_from_stage42_ef`
- role: turns the Stage42-EF source terms blocker into a fillable, auditable confirmation package.
- gate: `14 / 14`; verdict `stage42_eh_source_terms_confirmation_intake_pass`.
- intake template: `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`; schema: `outputs/stage42_long_research/source_terms_confirmation_schema_stage42.json`.
- top unblock targets: `['ucy_crowd_original', 'eth_biwi_original', 'aerialmpt_or_other_topdown']`; after-terms t50/t100 potential `10060` / `5696`.
- conversion_ready_now remains `0`; this stage does not download, convert, train, evaluate, or make metric/seconds claims.
<!-- STAGE42_EH_SOURCE_TERMS_CONFIRMATION_INTAKE:END -->

<!-- STAGE42_EI_SOURCE_TERMS_INTAKE_VALIDATOR_BRIDGE:START -->
## Stage42-EI Source Terms Intake Validator Bridge

- source: `fresh_validator_bridge_from_stage42_eh_intake`
- role: verifies that the CG validator now consumes the EH intake template and nested confirmation schema.
- gate: `10 / 10`; verdict `stage42_ei_intake_validator_bridge_pass`.
- validator_template_format: `stage42_eh_intake`; path `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`.
- conversion_ready_targets remains `0`; converted/evaluated now `0` / `0`.
- This fixes the EH->CG workflow bridge while preserving legal blocker, no metric/seconds claim, no Stage5C, and no SMC.
<!-- STAGE42_EI_SOURCE_TERMS_INTAKE_VALIDATOR_BRIDGE:END -->

<!-- STAGE42_EJ_GUARDED_SOURCE_CONVERSION_LAUNCHER:START -->
## Stage42-EJ Guarded Source Conversion Launcher

- source: `fresh_guarded_source_conversion_launcher_from_stage42_ei_manifest`
- role: reads the validator readiness manifest and creates a non-executing guarded conversion queue.
- gate: `12 / 12`; verdict `stage42_ej_guarded_source_conversion_launcher_pass`.
- ready targets: `0`; blocked targets: `5`; queued conversions: `0`.
- download/convert/evaluate executed: `False` / `False` / `False`.
- Current result preserves the legal blocker: no ready target means no conversion queue and no converted-data claim.
- Boundary: no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_EJ_GUARDED_SOURCE_CONVERSION_LAUNCHER:END -->

<!-- STAGE42_EK_LONG_OBJECTIVE_COVERAGE_AUDIT:START -->
## Stage42-EK Long Objective Coverage Audit

- source: `fresh_stage42_long_objective_coverage_audit`
- role: maps the active Stage42 A-F long objective to evidence rows, status labels, blockers, and paper-safe claims.
- gate: `10 / 10`; verdict `stage42_ek_long_objective_coverage_audit_pass_open_blockers`.
- requirements audited: `7` across phases `['A data and calibration', 'B external validation', 'C full-waypoint dynamics', 'D causal ablation', 'E safety floor', 'F paper package']`.
- paper files present: `9 / 9`.
- open blockers preserved: `['global_metric_seconds_claim_blocked', 'global_primary_full_waypoint_blocked', 'legal_conversion_ready_now_zero', 'neighbor_interaction_main_claim_blocked', 'scene_goal_main_claim_blocked', 'source_terms_confirmation_missing']`.
- completion/A-journal-ready claims remain disallowed; this is a coverage audit, not conversion/training/evaluation.
- Boundary: no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_EK_LONG_OBJECTIVE_COVERAGE_AUDIT:END -->

<!-- STAGE42_EL_CONTEXT_GAIN_ROUTER:START -->
## Stage42-EL Context Gain Router

- source: `fresh_stage42_context_gain_router`
- role: tests a deployment-aligned context target: supervised gain/harm routing over baseline-family protected control.
- gate: `10 / 10`; verdict `stage42_el_context_gain_router_pass`.
- positive_context_gain_routers: `[]`; best router `baseline_plus_history_goal_neighbor`.
- best all/t50/hard delta vs baseline-family: `0.000278` / `-0.000019` / `0.000321`; easy `-0.002666`.
- context_increment_verdict: `stage42_el_context_gain_router_not_supported`.
- Boundary: source-level raw-frame only; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_EL_CONTEXT_GAIN_ROUTER:END -->

<!-- STAGE42_EM_OFFICIAL_SOURCE_LINK_AUDIT:START -->
## Stage42-EM Official Source Link Audit

- source: `fresh_stage42_official_source_link_audit`
- role: record official source candidates and user confirmation blockers for the next guarded conversion.
- gate: `14 / 14`; verdict `stage42_em_official_source_link_audit_pass`.
- official/toolkit source candidates: `4` / `5`.
- conversion_ready_now: `0`; auto_download_allowed_now: `0`.
- estimated after-terms t50/t100 potential: `10060` / `5696`.
- No download, conversion, training, evaluation, metric/seconds claim, Stage5C, or SMC execution.
<!-- STAGE42_EM_OFFICIAL_SOURCE_LINK_AUDIT:END -->

<!-- STAGE42_EN_FLOOR_REMOVABILITY_DECISION_MAP:START -->
## Stage42-EN Floor Removability Decision Map

- source: `fresh_stage42_floor_removability_decision_map`
- role: maps which parts of Stage37/teacher floor can be removed, partially relaxed, or must remain.
- gate: `13 / 13`; verdict `stage42_en_floor_removability_decision_map_pass`.
- floor_free_neural_deployable: `False`; global_floor_removal_allowed: `False`.
- partial t50 relaxation available: `True`; teacher rollout context removal allowed: `False`.
- proximity guard required for safety-sensitive claim: `True`.
- Boundary: no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_EN_FLOOR_REMOVABILITY_DECISION_MAP:END -->

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

<!-- STAGE42_EP_DEPLOYMENT_CONTRACT_GUARD:START -->
## Stage42-EP Deployment Contract Guard

- source: `fresh_stage42_deployment_contract_guard`
- verdict: `stage42_ep_deployment_contract_guard_pass`
- gates: `16 / 16`
- role: machine-readable guard for deployment and paper-claim requests after Stage42-DN/EM/EN/EO.
- safety_sensitive_default: `proximity_guard`.
- source_level_runtime_candidate: `group_consistency_full_waypoint_runtime`.
- allowed only as diagnostic: `no_proximity_guard` accuracy-priority reporting.
- blocked: global floor-free neural deployment, teacher-floor rollout context removal, source conversion without user terms, metric/seconds/foundation claims, Stage5C execution, and SMC.
- unknown future policy requests are denied by default until explicitly added to the contract.
<!-- STAGE42_EP_DEPLOYMENT_CONTRACT_GUARD:END -->

<!-- STAGE42_EQ_SEQUENCE_GRAPH_CONTEXT_ROUTER:START -->
## Stage42-EQ Sequence+Graph Context Router

- source: `fresh_stage42_sequence_graph_context_router`
- role: tests whether past-only sequence summary + current-frame graph summary can improve context gain routing over baseline-family protected control.
- gate: `12 / 12`; verdict `stage42_eq_sequence_graph_context_router_pass`.
- positive_sequence_graph_context_routers: `[]`; best router `baseline_plus_history_goal_neighbor`.
- best all/t50/t100raw/hard delta vs baseline-family: `0.000118` / `-0.000197` / `0.000083` / `0.000169`; easy `-0.001971`.
- sequence_graph_increment_verdict: `stage42_eq_sequence_graph_context_router_not_supported`.
- Boundary: fresh router audit only; raw-frame/dataset-local 2.5D; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_EQ_SEQUENCE_GRAPH_CONTEXT_ROUTER:END -->

<!-- STAGE42_ER_POST_EQ_CONTEXT_CLAIM_REFRESH:START -->
## Stage42-ER Post-EQ Context Claim Refresh

- source: `fresh_post_eq_context_claim_refresh`
- role: updates paper/action boundaries after the fresh Stage42-EQ sequence+graph router result.
- gate: `14 / 14`; verdict `stage42_er_post_eq_context_claim_refresh_pass`.
- Stage42-EQ best all/t50/t100raw/hard delta: `0.01%` / `-0.02%` / `0.01%` / `0.02%`.
- context decision: `close_current_shallow_sequence_graph_context_protocol`; independent context main claim allowed `False`.
- DA-2 is closed negative under the current shallow sequence/graph residual/router protocols.
- New priority: source/legal/time conversion plus stronger joint occupancy or interaction-constraint targets.
- Boundary: raw-frame/dataset-local 2.5D only; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_ER_POST_EQ_CONTEXT_CLAIM_REFRESH:END -->

<!-- STAGE42_ES_INTERACTION_OCCUPANCY_TARGET_SELECTION:START -->
## Stage42-ES Interaction / Occupancy Target Selection

- source: `fresh_stage42_interaction_occupancy_target_selection`
- role: fresh-reruns DH scalar proximity/occupancy target and DI explicit group-consistency target to choose the next interaction/occupancy training route.
- gate: `17 / 17`; verdict `stage42_es_interaction_occupancy_target_selection_pass`.
- selected target family: `explicit_group_consistency_repair`; decision `continue_with_explicit_group_consistency_interaction_target`.
- selected group-consistency all/t50/t100raw/hard/easy: `24.72%` / `22.36%` / `14.35%` / `23.89%` / `-25.63%`.
- near@0.05 base/final: `1.94%` / `1.38%`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_ES_INTERACTION_OCCUPANCY_TARGET_SELECTION:END -->

<!-- STAGE42_ET_GROUP_CONSISTENCY_TARGET_ABLATION:START -->
## Stage42-ET Group-Consistency Target Ablation

- source: `fresh_stage42_group_consistency_target_ablation`
- role: tests whether the Stage42-ES selected interaction/occupancy target depends on real source/frame/horizon multi-agent grouping.
- gate: `16 / 16`; verdict `stage42_et_group_consistency_target_ablation_pass`.
- selected target for next stage: `source_frame_horizon`; decision `keep_source_frame_horizon_group_consistency_target`.
- source/frame/horizon all/t50/t100raw/hard/easy: `24.72%` / `22.36%` / `14.35%` / `23.89%` / `-25.63%`.
- agent-isolated control all/t50/hard/easy: `24.58%` / `22.02%` / `23.75%` / `-25.66%`.
- hard increment vs isolated `0.14%`; own-base near@0.05 reduction `0.55%`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_ET_GROUP_CONSISTENCY_TARGET_ABLATION:END -->

<!-- STAGE42_EU_GROUP_CONSISTENCY_CONSTRAINT_TRAINING:START -->
## Stage42-EU Group-Consistency Constraint Training

- source: `fresh_stage42_group_consistency_constraint_training`
- role: trains source/frame/horizon group-risk weighted full-waypoint dynamics, then applies validation-selected group repair.
- gate: `15 / 18`; verdict `stage42_eu_group_consistency_constraint_training_positive_not_promoted`.
- selected training variant: `group_unsafe_weighted` with lambda `10.0`.
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
- gate: `12 / 14`; verdict `stage42_ev_constraint_aware_composer_positive_not_promoted`.
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
- role: validation-only adaptive repair over Stage42-DI candidate grid by global / domain+horizon / domain+horizon+risk slices.
- gate: `14 / 16`; verdict `stage42_ew_adaptive_group_repair_positive_not_promoted`.
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
- role: validation-only adaptive repair with risk aggregated to source/frame/horizon groups before candidate selection.
- gate: `15 / 17`; verdict `stage42_ex_group_level_risk_repair_positive_not_promoted`.
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
- gate: `16 / 18`; verdict `stage42_ey_continuous_group_risk_repair_positive_not_promoted`.
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

<!-- STAGE42_FJ_FH_SOURCE_ROBUSTNESS:START -->
## Stage42-FJ FH Source / Domain / Horizon Robustness Audit

- source: `fresh_stage42_fh_source_robustness_audit`
- role: audit frozen Stage42-FH/FI policy across domain/source/horizon/scene slices without retraining or threshold reselection.
- gate: `14 / 14`; verdict `stage42_fj_fh_source_robustness_pass`.
- robust domains: `['TrajNet', 'UCY']`.
- weak domains: `[]`.
- robust domain-horizon slices: `['TrajNet|10', 'TrajNet|25', 'TrajNet|50', 'UCY|10', 'UCY|25']`.
- weak domain-horizon slices: `['TrajNet|100', 'UCY|50', 'UCY|100']`.
- robust sources: `['TrajNet/Test/crowds/students002.txt', 'TrajNet/Train/crowds/crowds_zara03.txt', 'TrajNet/Train/crowds/students003.txt']`.
- weak sources: `[]`.
- dual-domain positive-safe claim allowed: `True`.
- broad uniform source claim allowed: `True`.
- broad uniform horizon claim allowed: `False`.
- Boundary: frozen protected source-level raw-frame 2.5D audit; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FJ_FH_SOURCE_ROBUSTNESS:END -->

<!-- STAGE42_FK_FH_HORIZON_WEAK_SLICE_REPAIR:START -->
## Stage42-FK FH Horizon Weak-Slice Validation Repair

- source: `fresh_stage42_fh_horizon_weak_slice_repair`
- role: validation-only repair attempt for FJ weak horizon slices; no retraining and no test threshold tuning.
- gate: `15 / 15`; verdict `stage42_fk_fh_horizon_weak_slice_repair_pass_with_horizon_limit`.
- global all/t50/t100raw/hard/easy: `35.18%` / `28.97%` / `21.13%` / `33.33%` / `-36.88%`.
- weak horizons before: `['TrajNet|100', 'UCY|50', 'UCY|100']`.
- weak horizons after: `['TrajNet|100', 'UCY|50', 'UCY|100']`.
- applied overrides: `{'TrajNet|100': {'candidate': 'fb', 'rows': 5608, 'reason': 'validation_safe_best_score'}, 'UCY|50': {'candidate': 'fh', 'rows': 2340, 'reason': 'validation_safe_best_score'}, 'UCY|100': {'candidate': 'fa', 'rows': 1440, 'reason': 'validation_safe_best_score'}}`.
- uniform horizon claim allowed: `False`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FK_FH_HORIZON_WEAK_SLICE_REPAIR:END -->

<!-- STAGE42_FL_FH_HORIZON_WEAK_SLICE_FORENSICS:START -->
## Stage42-FL FH Weak-Horizon Forensics

- source: `fresh_stage42_fh_horizon_weak_slice_forensics`
- role: fresh diagnostic for FK/FJ weak horizons; no policy promotion and no test threshold tuning.
- gate: `15 / 15`; verdict `stage42_fl_horizon_weak_slice_forensics_pass`.
- analyzed weak horizons: `['TrajNet|100', 'UCY|50', 'UCY|100']`.
- root cause counts: `{'oracle_label_low_margin_ambiguous': 3}`.
- next action: `train_horizon_specific_row_level_switch_model_with_stronger_history_neighbor_goal_features`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC; uniform horizon claim still blocked.
<!-- STAGE42_FL_FH_HORIZON_WEAK_SLICE_FORENSICS:END -->

<!-- STAGE42_FM_FH_HORIZON_ROW_SWITCH_SPECIALIST:START -->
## Stage42-FM FH Weak-Horizon Row-Level Switch Specialist

- source: `fresh_stage42_fh_horizon_row_switch_specialist`
- role: validation-only row-level specialist attempt for FK/FJ/FL weak horizon slices; no test threshold tuning.
- gate: `15 / 15`; verdict `stage42_fm_horizon_row_switch_specialist_pass_with_horizon_limit`.
- global all/t50/t100raw/hard/easy: `35.20%` / `29.03%` / `21.14%` / `33.35%` / `-37.10%`.
- weak horizons before: `['TrajNet|100', 'UCY|50', 'UCY|100']`.
- weak horizons after: `['TrajNet|100', 'UCY|100']`.
- applied policies: `{'TrajNet|100': {'key': 'TrajNet|100', 'mode': 'feature_threshold', 'candidate': 'fb', 'feature': 'path_length', 'direction': 'ge', 'threshold': 0.3749999749633932, 'rows': 5608, 'switch_rows': 3008}, 'UCY|50': {'key': 'UCY|50', 'mode': 'feature_threshold', 'candidate': 'di', 'feature': 'endpoint_delta_fh', 'direction': 'le', 'threshold': 0.026976035023941254, 'rows': 2340, 'switch_rows': 1170}, 'UCY|100': {'key': 'UCY|100', 'mode': 'feature_threshold', 'candidate': 'fb', 'feature': 'endpoint_delta_floor', 'direction': 'ge', 'threshold': 0.02336742544527692, 'rows': 1440, 'switch_rows': 936}}`.
- uniform horizon claim allowed: `False`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FM_FH_HORIZON_ROW_SWITCH_SPECIALIST:END -->

<!-- STAGE42_FN_FH_HORIZON_CONSERVATIVE_EASY_GUARD:START -->
## Stage42-FN FH Horizon Conservative Easy Guard

- source: `fresh_stage42_fh_horizon_conservative_easy_guard`
- role: validation-only conservative easy-safety guard for FM remaining weak horizon slices; no test threshold tuning.
- gate: `15 / 15`; verdict `stage42_fn_conservative_easy_guard_pass_with_horizon_limit`.
- global all/t50/t100raw/hard/easy: `34.86%` / `29.03%` / `20.19%` / `32.96%` / `-37.14%`.
- weak horizons before: `['TrajNet|100', 'UCY|100']`.
- weak horizons after: `['TrajNet|100', 'UCY|100']`.
- applied guards: `{'TrajNet|100': {'key': 'TrajNet|100', 'mode': 'feature_guard', 'replacement': 'floor', 'feature': 'path_length', 'direction': 'le', 'threshold': 0.3749999749633932, 'rows': 5608, 'guard_rows': 2593}, 'UCY|100': {'key': 'UCY|100', 'mode': 'feature_guard', 'replacement': 'fa', 'feature': 'min_distance', 'direction': 'le', 'threshold': 0.12583341276755197, 'rows': 1440, 'guard_rows': 288}}`.
- uniform horizon claim allowed: `False`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FN_FH_HORIZON_CONSERVATIVE_EASY_GUARD:END -->

<!-- STAGE42_FO_FH_HORIZON_GAIN_HARM_SPECIALIST:START -->
## Stage42-FO FH Horizon Gain/Harm Specialist

- source: `fresh_stage42_fh_horizon_gain_harm_specialist`
- role: validation-only row-level gain/harm specialist for remaining weak horizon slices; no test threshold tuning.
- gate: `16 / 16`; verdict `stage42_fo_gain_harm_specialist_pass_with_horizon_limit`.
- global all/t50/t100raw/hard/easy: `35.20%` / `29.03%` / `21.14%` / `33.35%` / `-37.10%`.
- weak horizons before: `['TrajNet|100', 'UCY|100']`.
- weak horizons after: `['TrajNet|100', 'UCY|100']`.
- applied policies: `{'TrajNet|100': {'key': 'TrajNet|100', 'mode': 'gain_harm_model', 'gain_min': 0.0, 'harm_max': 0.35, 'max_switch': 0.35, 'rows': 5608, 'switch_rows': 1962}, 'UCY|100': {'key': 'UCY|100', 'mode': 'keep_fm', 'rows': 1440, 'switch_rows': 0}}`.
- uniform horizon claim allowed: `False`.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_FO_FH_HORIZON_GAIN_HARM_SPECIALIST:END -->

<!-- M3W_GOAL_EVIDENCE_LEDGER_REFRESH:START -->
## M3W Goal Evidence Ledger Refresh

- source: `cached_verified_summary_from_stage18_to_stage42_fo_reports`
- role: user-requested single-file Chinese summary of what was attempted under the M3W long-term goal, which routes failed and why, which routes succeeded, current model quality, current best deployable policies, claim boundaries, and next actions.
- canonical README: `README_M3W_GOAL_EVIDENCE_LEDGER_ZH.md`.
- current positioning: protected dataset-local/raw-frame 2.5D multi-agent world-state candidate.
- current best deployable families: Stage26 for SDD pixel raw-frame; Stage37 for external t50 selective transfer; Stage42-FH/FI family for frozen source/domain robust protected policy.
- current blocker: uniform horizon robustness is still blocked by `TrajNet|100` and `UCY|100`; Stage42-FN/FO did not repair these h100 weak slices.
- not claimed: true 3D, foundation model, global metric prediction, seconds-level horizon, ungated neural dynamics deployment, Stage5C execution, or SMC readiness.
- This is a documentation/evidence-ledger refresh, not a new training run.
- verification: `.venv-pytorch/bin/python -m json.tool research_state.json` passed; `.venv-pytorch/bin/python -m pytest tests` -> `828 passed in 31.08s`.
<!-- M3W_GOAL_EVIDENCE_LEDGER_REFRESH:END -->

<!-- STAGE42_FP_H100_WEAK_HORIZON_SOURCE_SUPPORT_AUDIT:START -->
## Stage42-FP H100 Weak-Horizon Source / Support Audit

- source: `fresh_stage42_h100_weak_horizon_source_support_audit`
- role: diagnostic source/support decomposition for remaining h100 weak horizons after Stage42-FO; no new training and no test threshold tuning.
- gate: `15 / 15`; verdict `stage42_fp_h100_source_support_audit_pass`.
- h100 weak horizons: `['TrajNet|100', 'UCY|100']`.
- blocker counts: `{'long_horizon_h100_context_still_insufficient': 2, 'low_material_headroom': 2, 'oracle_low_margin_ambiguous': 2, 'single_or_sparse_validation_source_support': 2, 'source_specific_easy_safety_ci_failure': 2, 'validation_to_test_source_family_shift': 2, 'gain_harm_policy_abstained_due_to_validation_safety': 1}`.
- recommended next action: `source_support_or_long_horizon_context_repair_before_retrying_policy_promotion`.
- conclusion: uniform horizon robustness remains blocked; TrajNet|100 and UCY|100 need source/support or stronger long-horizon context repair before any policy promotion.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
- verification: `.venv-pytorch/bin/python run_stage42_h100_weak_horizon_source_support_audit.py` -> `15 / 15`; focused pytest `4 passed`; full pytest `832 passed in 30.13s`.
<!-- STAGE42_FP_H100_WEAK_HORIZON_SOURCE_SUPPORT_AUDIT:END -->
