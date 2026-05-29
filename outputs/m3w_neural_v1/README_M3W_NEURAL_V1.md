# M3W-Neural v1

M3W-Neural v1 is a Stage41-protected neural world-dynamics candidate. It combines composite-tail bounded neural dynamics with a validation-selected safe-switch policy and the Stage37/teacher safety floor.

It is not true 3D, not metric, not seconds-level, not a foundation model, and not Stage5C/SMC.

Current user-facing master summary:

`/Users/yangyue/Downloads/World/README_M3W_CURRENT_MASTER_SUMMARY_ZH.md`

This summary consolidates the goal-level routes tried, failed routes and causes, successful evidence, current best deployable hierarchy, and strict claim boundaries. It is `cached_verified` summary-only and does not introduce new training, conversion, download, or evaluation.

Latest Stage42-GT floor-relaxation safety stress test:

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/floor_relaxation_safety_stress_stage42.md`

Stage42-GT is a `fresh_run` all-agent safety stress test for the Stage42-BY/BZ validation-backed t50 floor-relaxation policy. It replays the alpha-blended policy, groups rows by `source_file + frame_id + horizon`, and compares partial relaxation against the teacher/floor rollout. Target union t50 rows = `11538`; t50 improvement = `+28.97%`; hard/failure = `+28.97%`; easy degradation = `-21.41%`; near-collision@0.05 delta = `-0.74pp`; jagged-rate delta = `0.00pp`; gate = `14 / 14`. This supports narrow validation-backed t50 floor relaxation, not global floor removal or floor-free neural deployment.

Latest single-file Chinese work ledger requested by the user:

`/Users/yangyue/Downloads/World/README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`

This README summarizes, in one place, what was attempted under the M3W long goal, which routes failed and why, which routes succeeded, current model quality, current best deployable families, strict claim boundaries, and next actions. It now includes Stage42-ES through Stage42-FP: interaction/occupancy scalar targets remain diagnostic, explicit source/frame/horizon group-consistency is the supported target family, later group-risk/repel/Pareto repairs did not become a new best deployable policy, objective-level proximity training improved all/t50/hard while still failing the proximity safety gate, FA safety-teacher target blending was selected away by validation, Stage42-FE constrained FC-to-DI safety fallback restored proximity safety while preserving FC-level all/t50/hard gains, Stage42-FH repaired the UCY weak-domain problem with train-only internal validation, Stage42-FI froze/replayed that FH policy with exact replay plus 2000-bootstrap evidence, Stage42-FJ confirms dual-domain/source robustness while blocking uniform horizon overclaim, Stage42-FK attempts validation-only horizon repair but still keeps the uniform-horizon claim blocked, Stage42-FL diagnoses the remaining weak horizons as low-margin ambiguous slices, Stage42-FM shows a row-level horizon specialist can repair UCY|50 and improve global all/t50/t100raw/hard to 35.20% / 29.03% / 21.14% / 33.35% while still leaving TrajNet|100 and UCY|100 as horizon-specific safety blockers, Stage42-FN shows that adding a stricter conservative easy guard keeps global safety but does not repair those remaining two weak horizons, Stage42-FO shows that a validation-only gain/harm specialist using Stage37/past/prototype/rollout features still cannot unlock the remaining h100 weak slices, and Stage42-FP decomposes those h100 blockers into source/support/context causes rather than treating them as another global-threshold problem.

Canonical concise Chinese evidence ledger:

`/Users/yangyue/Downloads/World/README_M3W_GOAL_EVIDENCE_LEDGER_ZH.md`

This is the current user-facing single-file summary of attempted routes, failed routes and causes, successful evidence, best deployable policies, strict claim boundaries, and the remaining h100 weak-horizon blocker.

Latest current-goal work summary requested by the user:

`/Users/yangyue/Downloads/World/README_M3W_TARGET_WORK_SUMMARY_ZH.md`

This root README summarizes what was attempted under the M3W long goal, which routes failed and why, which routes succeeded, the current best deployable family, and the strict claim boundaries. It now includes the latest Stage42-EE/EF/EG evidence: context switchability remains below materiality threshold, source conversion remains legally blocked until terms/path confirmation, and the paper claim matrix only allows the protected source-level group-consistency full-waypoint claim.

Latest user-facing route/failure/success summary:

`/Users/yangyue/Downloads/World/README_M3W_DETAILED_RESULTS_ZH.md`

This is the current concise-but-detailed Chinese summary requested by the user. It covers attempted routes, failed routes and causes, successful evidence, current best deployable status, claim boundaries, and the Stage42-BD local t100 source inventory.

Latest Stage42 goal/scene gated expert audit:

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/goal_scene_gated_expert_stage42.md`

Stage42-CJ passes `10 / 10` gates as `fresh_run`, but it is diagnostic rather than a new positive contribution. The validation-only gate selected `baseline_family_control`, not the goal/scene candidates. Test metrics confirm the boundary: `baseline_family_control` all/t50/hard = `28.78% / 31.54% / 27.58%`, while `baseline_plus_goal_scene` drops to `26.25% / 22.76% / 24.86%` and `baseline_plus_motion_goal_context` drops to `24.58% / 22.02% / 23.75%`. Therefore goal/scene remains mixed/diagnostic and must not be written as an independent main claim under the current source-level ridge/full-waypoint protocol.

Latest Stage42 neighbor/interaction gated expert audit:

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/neighbor_interaction_gated_expert_stage42.md`

Stage42-CK passes `11 / 11` gates as `fresh_run`, but it is also diagnostic rather than a new positive contribution. It builds current-frame kNN graph features for `337991` rows (`334525` rows with neighbors), then evaluates scalar-neighbor and graph candidates under a validation-only safe gate. The gate again selects `baseline_family_control`: scalar neighbor reaches `26.37% / 22.96% / 24.88%` all/t50/hard, kNN graph reaches `24.38% / 22.38% / 23.78%`, both below the baseline-family control `28.78% / 31.54% / 27.58%`. Neighbor/interaction therefore remains auxiliary/diagnostic and not an independent main claim under the current source-level ridge/full-waypoint protocol.

Latest Stage42 post-context-guard paper package refresh:

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/paper_package_post_context_guard_refresh_stage42.md`

Stage42-CL passes `11 / 11` gates as `fresh_synthesis_from_stage42_cj_ck_artifacts`. It propagates the CJ/CK negative evidence into the paper package files themselves: experiment tables, ablation tables, failure taxonomy, and A-journal gap. The refreshed package explicitly blocks goal/scene and neighbor/interaction as independent main claims while preserving the supported mechanism boundary: baseline-family rollout context, causal history, guarded domain expert, and conservative safety floor. This is a claim-boundary refresh, not new training, not metric/seconds-level evidence, not Stage5C, and not SMC.

Latest Stage42 endpoint bridge / full-waypoint shape audit:

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/full_waypoint_bridge_shape_audit_stage42.md`

Stage42-CM passes `14 / 14` gates as `fresh_synthesis_from_stage42_full_waypoint_artifacts`. It clarifies a key world-state boundary: protected full-waypoint sequence dynamics improves t50 by `+1.15%` and t100 raw-frame diagnostic by `+8.16%` relative to the composite-tail endpoint-linear bridge, but loses `-2.45%` on all-ADE and `-0.87%` on hard/failure. Endpoint-linear bridge remains the stronger all-ADE floor; endpoint-only success cannot be counted as learned full-waypoint dynamics. Ungated full-waypoint neural still has unsafe easy degradation (`124.59%`), so deployment remains protected by Stage37/teacher floor and safe switch.

Latest Stage42 bridge / shape composer audit:

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/bridge_shape_composer_stage42.md`

Stage42-CN passes `15 / 15` gates as `fresh_synthesis_from_stage42_cm_j_x_artifacts`, but it documents a blocker rather than a new deployment switch. Stage42-J static-gated full-waypoint evidence is validation-selected and easy-safe, and Stage42-CM shows full-waypoint has auxiliary t50/t100 raw-frame lift. However, there is still no common validation-aligned endpoint-linear-vs-full-waypoint row cache, so a bridge/shape composer cannot safely switch deployment policy yet. The deployable policy remains endpoint-linear bridge / Stage37-teacher floor, with full-waypoint reported as auxiliary horizon evidence.

Latest Stage42 common-validation bridge / shape composer:

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/common_validation_bridge_shape_composer_stage42.md`

Stage42-CO passes `14 / 14` gates as `fresh_common_validation_eval_from_cached_verified_checkpoints`. It resolves the Stage42-CN blocker by verifying exact validation/test row alignment between endpoint-linear bridge and full-waypoint sequence. The validation-only composer selects full-waypoint only for `ETH_UCY|50` and `ETH_UCY|100`, then evaluates test once. Test ADE improves over endpoint-linear bridge by `+3.02%` all, `+1.50%` t50, `+6.12%` t100 raw-frame diagnostic, and `+3.28%` hard/failure, with easy degradation `+0.25%`. The result is a protected bridge/shape improvement, still dataset-local/raw-frame 2.5D and not metric/seconds-level, Stage5C, or SMC evidence.

Latest Stage42 common-validation composer safety / bootstrap audit:

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/common_validation_composer_safety_stage42.md`

Stage42-CP passes `14 / 14` gates as `fresh_joint_safety_bootstrap_from_stage42_co_policy`. It adds `2000`-bootstrap statistical evidence and all-agent joint safety checks to the Stage42-CO composer. Against the endpoint-linear bridge, test ADE improves by `+3.02%` all with CI `[+2.64%, +3.37%]`, `+1.50%` t50 with CI `[+0.90%, +2.09%]`, `+6.12%` t100 raw-frame diagnostic with CI `[+5.39%, +6.94%]`, and `+3.28%` hard/failure with CI `[+2.90%, +3.68%]`. The safety caveat is explicit: near-collision@0.05 is `+0.34%` versus endpoint-linear but `-0.05%` versus the strongest floor, and jagged-rate does not worsen. This is protected dataset-local/raw-frame 2.5D evidence, not metric/seconds-level, Stage5C, or SMC evidence.

Latest Stage42 proximity-aware composer guard:

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/proximity_aware_composer_guard_stage42.md`

Stage42-CQ passes `19 / 19` gates as `fresh_validation_selected_proximity_guard_from_stage42_co_policy`. It directly repairs the Stage42-CP safety caveat by selecting a predicted-proximity guard on validation only (`min_sep=0.2`, `margin=0.005`) and evaluating test once. Against endpoint-linear bridge, the guarded composer keeps positive ADE gains: all `+1.77%`, t50 `+1.07%`, t100 raw-frame diagnostic `+3.48%`, hard/failure `+1.93%`, easy degradation `+0.25%`; bootstrap lower bounds are positive for all and t50. Near-collision@0.05 becomes `-0.06%` versus endpoint-linear and `-0.45%` versus the strongest floor. This is a safer protected composer variant, not metric/seconds-level, Stage5C, or SMC evidence.

Latest Stage42 proximity guard ablation / Pareto audit:

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/proximity_guard_ablation_stage42.md`

Stage42-CR passes `19 / 19` gates as `fresh_synthesis_from_stage42_co_cp_cq_artifacts`. It makes the safety/accuracy tradeoff explicit: the no-guard composer has higher ADE gain (`+3.02%` all, `+1.50%` t50, `+6.12%` t100 raw, `+3.28%` hard/failure) but worsens near-collision@0.05 by `+0.34%` versus endpoint-linear. The guarded composer gives up accuracy (`+1.77%` all, `+1.07%` t50, `+3.48%` t100 raw, `+1.93%` hard/failure) but repairs near-collision@0.05 to `-0.06%`. The recommended safety-sensitive policy is therefore the proximity guard; the no-guard variant is diagnostic/accuracy-priority only.

Latest Stage42 safety-floor audit:

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/safety_floor_necessity_audit_stage42.md`

Stage42-BW passes `15 / 15` gates as `fresh_stage42_bw_safety_floor_necessity_audit`. It separates fallback-floor deployment safety from teacher/floor rollout context and from ungated neural dynamics. The current protected composite-tail policy remains positive and easy-safe (`all 21.03%`, `t50 13.65%`, `hard 20.38%`, `easy 0.00%`). Ungated endpoint/full-waypoint variants show unacceptable easy degradation (`124.59%`), and removing floor/safe rollout context hurts protected t50 (`-9.21%` and `-9.50%`). Therefore Stage37/teacher floor remains a necessary deployment safety mechanism and baseline-family rollout context remains the dominant supported mechanism; this is not true 3D, not metric/seconds-level, not Stage5C, and not SMC evidence. Verification: focused pytest `5 passed`; full pytest `502 passed in 68.35s`.

Latest Stage42 slice-level floor-relaxability audit:

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/floor_relaxability_audit_stage42.md`

Stage42-BX passes `14 / 14` gates as `fresh_stage42_bx_floor_relaxability_audit`. It shows fallback relaxation is slice-limited, not global: only `TrajNet|25` is relaxable under validation and final-test safety rules. `TrajNet|50` is blocked by validation easy harm, `UCY|50` is blocked by missing validation support, and no t100 slice is relaxable. The teacher/floor rollout context and Stage37 safety floor therefore remain required for deployment. Verification: focused pytest `8 passed`; full pytest `507 passed in 67.02s`.

Previous long-form research ledger:

`/Users/yangyue/Downloads/World/README_M3W_RESEARCH_SUMMARY_ZH.md`

That README now starts with a direct “本次交付版总摘要” covering what was attempted, which routes failed and why, which routes succeeded, the current best deployable model, and the strict claim boundary. It keeps the core limitation explicit: M3W-Neural v1 is still a protected dataset-local raw-frame 2.5D multi-agent world-state candidate, not true 3D, not foundation-scale, not metric/seconds-level, not Stage5C, and not SMC.

## Files

- `README_GOAL_SUMMARY_M3W_NEURAL_V1.md` — detailed research ledger: attempted routes, failures, successes, current best deployable candidate, and remaining gaps.
- `README_M3W_GOAL_DETAILED_SUMMARY_ZH.md` — Chinese goal-level README for the full M3W route, including failed paths, successful paths, claim boundaries, and Stage42-R row-cache combo status.
- `report_m3w_neural_v1.md` — frozen result summary.
- `evidence_matrix_m3w_neural_v1.md/json` — gate and metric evidence.
- `selector_policy_m3w_neural_v1.json` — frozen policy metadata and hashes.
- `model_card_m3w_neural_v1.md` — intended use and limitations.
- `data_card_m3w_neural_v1.md` — dataset and leakage status.
- `reproducibility_m3w_neural_v1.md` — rerun commands.
- `paper_gap_m3w_neural_v1.md` — what is still missing before stronger publication claims.

Latest package inputs include the negative fixed-composer source-switch audits and the positive strict pure-UCY neural retrain/statistical evidence, so the frozen package records both the successful composite-tail path and the repaired source-only neural branch.

The package also includes the positive endpoint-to-full bridge audit: domain-local endpoint neural dynamics pass actual full-waypoint ADE/FDE, multi-agent, proximity, and smoothness gates on ETH_UCY and TrajNet through a linear waypoint bridge. This strengthens world-state evidence without claiming learned waypoint-shape dynamics.

The endpoint-to-full bridge now also has fresh 2000-bootstrap per-domain statistical support on ETH_UCY and TrajNet. The lower bounds are positive for all/t50/hard/multi-agent ADE and all/t50 FDE, but this is still protected linear-bridge evidence rather than ungated learned full-waypoint shape dynamics.

The required ablation coverage audit is now packaged. It covers no-history, no-neighbor, no-scene/goal, no-interaction, no-JEPA, no-Transformer, and no-fallback. The newer same-protocol neural architecture audit records that pure Transformer/no-JEPA, JEPA-only/no-Transformer, and JEPA+Transformer hybrid attempts were negative or fallback-only under the Stage41 external protocol.

The package includes a calibrated learned-shape meta-policy as well. It selects protected waypoint-shape residual sources on validation, evaluates test once, and remains positive on ETH_UCY and TrajNet. The learned-shape contribution is small and protected, not an ungated neural replacement.

The Stage42-AE row-cache stress audit is now packaged too. It confirms Stage42-X global t50 remains seed/bootstrap positive, but it records limitations rather than overclaiming: ETH_UCY has weak t50/FDE@50 lower bounds and horizon=25 is not uniformly positive. Stage42-AF repairs the horizon=25 weak slice with a validation-only low-margin guard. Stage42-AG then repairs the ETH_UCY t50/FDE@50 lower-bound weakness with a validation-only FDE-aware source guard. Stage42-AH refreshes the post-repair paper-claim boundary. Stage42-AI repairs the remaining TrajNet|100 easy-safety limit with a validation-only source guard while keeping t100 raw-frame diagnostic only. Stage42-AJ propagates AD-AI evidence into all Stage42 paper package files.

Stage42-AL and Stage42-AM refine the source-level claim boundary. Stage42-AL shows that the locked post-repair stress pool is not a full proposed source-level split evaluation. Stage42-AM then evaluates the proposed source-level test split directly with a fresh past-only ridge full-waypoint probe and validation-only safe policy: test rows `47458`, TrajNet `37918`, UCY `9540`, ADE all `+0.245788`, t50 `+0.220171`, t100 raw-frame diagnostic `+0.143652`, hard/failure `+0.237494`, easy degradation `-0.256627`, gates `12 / 12`. This is strong source-level raw-frame full-waypoint evidence, but it is still a protected dataset-local 2.5D probe, not metric, seconds-level, true 3D, foundation, Stage5C, or SMC evidence.

Stage42-AN adds a retrained source-level ablation boundary. It reruns ridge probes and validation-only safety policies for no-history, no-neighbor, no-goal, no-baseline-family, no-domain, and combined variants on the same proposed source-level split. The full variant remains strong, but the gate is `9 / 10`: only `baseline_family_context` is independently supported. History, neighbor/interaction, goal prototype, domain expert, and safe-switch necessity are not proven by this ridge ablation, so they must not be written as independent main claims yet.

Stage42-AO sharpens that boundary with standalone and incremental variants. `history_only` and `motion_goal_context` show standalone positive signal, but no context variant improves over `baseline_family_only`; `baseline_family_only` is stronger than the full ridge variant on all/t50/hard. The current source-level ridge evidence is therefore dominated by baseline-family rollout context. History, goal, neighbor, and domain modules need stronger neural/graph retraining or richer context before becoming independent paper claims.

Stage42-AP then tests the same question as a two-stage residual problem. It trains a baseline-family first-stage model and asks history / goal / neighbor context to predict the remaining full-waypoint residual. The result is still partial/negative: no residual context variant improves over the baseline-family first stage by the +1% threshold. This further constrains paper claims: current source-level ridge/residual evidence supports baseline-family rollout context, not independent history/goal/neighbor contribution.

Stage42-AQ repeats the residual-context question with a real PyTorch MLP in the arm64 `.venv-pytorch` runtime. It still does not find neural context increment: `neural_history`, `neural_goal_neighbor`, and `neural_history_goal_neighbor` all underperform the baseline-family first-stage on all/t50/hard. This rules out a simple tabular neural-context repair; the next credible route is graph/sequence/scene-rich context.

Stage42-AR then tests temporal sequence context directly using a Conv1D encoder over past-only `history_seq` with shape `[337991, 64, 7]`. It also fails to beat the baseline-family first-stage: sequence-history and sequence-goal/neighbor variants underperform on all/t50/hard. The evidence now strongly says the current source-level ridge/residual/tabular/sequence context branches do not independently explain the gain beyond baseline-family rollout context.

Stage42-AS adds a structured current-frame kNN graph / interaction-context residual test. It builds graph features for `337991` rows, with `334525` rows having same-frame neighbors and up to `65` unique agents per frame, using only current/past motion and no future labels as input. The result is still partial/negative: `graph_only`, `graph_goal`, and `graph_history_goal` underperform the baseline-family first-stage on all/t50/hard. This rules out the current hand-built graph residual context as an independent source-level contribution and further supports treating baseline-family rollout context as the dominant current mechanism unless a stronger graph-neural or scene-token protocol proves otherwise.

Stage42-AT audits the safety floor boundary on the same proposed source-level split. It finds that fallback removal is supported for the baseline-family ridge probe itself: ungated all-row prediction reaches all `+0.461656`, t50 `+0.411874`, hard/failure `+0.458447`, with easy degradation still negative. But this is not a floor-free neural claim, because baseline/floor rollout context remains an input mechanism. Removing floor/safe rollout context hurts protected t50, so teacher/floor context removal is not supported as a global replacement.

Stage42-AU decomposes that baseline-family mechanism. Horizon/domain controls alone do not work; `floor_rel_only` is weak; `safe_baseline_rel_only` is unsafe for t50; `family_baseline_rel_only` is the dominant single source-level mechanism, with protected all `+0.273815` and t50 `+0.237296`. The full `baseline_family_all` context improves protected t50 to `+0.315425`. The current mechanism claim should therefore focus on baseline-family rollout context, especially family relative rollout, while not overclaiming independent history/goal/neighbor/sequence/graph contributions.

Stage42-AV checks robustness and weak slices for that mechanism. The global bootstrap lower bounds are strongly positive for `baseline_family_all` (all `+0.284243`, t50 `+0.309806`, hard/failure `+0.271961`, easy-degradation high `-0.459376`), and TrajNet is positive. But UCY has no validation rows in this proposed source-level split and is therefore floor-only, not positive transfer. Horizon 100 remains raw-frame diagnostic with an easy-safety weak slice. Uniform domain/horizon claims are still disallowed.

Stage42-AW repairs that UCY validation-support blocker. It carves `UCY::UCY/zara03/crowds_zara03.txt` from original UCY train sources as internal validation, keeps test sources unchanged, and selects policies without test metrics. The validation-best `family_baseline_rel_only` variant reaches global all `+0.356806`, t50 `+0.289698`, hard/failure `+0.338904`; UCY test all `+0.374492`, t50 `+0.245320`, hard/failure `+0.355073`, with negative easy degradation. This is a repaired validation-support protocol, not metric/seconds-level or true-3D evidence.

Stage42-BB packages the current t100 limitation as an actionable data/calibration gap. It reads Stage42-BA train-only source-CV support and the Stage42 calibration audit, then reports that no external domain currently has enough independent t100 support: ETH_UCY needs at least 2 additional safe t100-capable train sources or source-specific repair, TrajNet needs at least 1, and UCY needs at least 1 more t100-capable original-train source. After the source-CV guard, all/t50/hard remain positive and easy remains safe, but t100 raw-frame diagnostic remains `0.0`. The user-action file is `outputs/stage42_long_research/user_action_required_t100_stage42.md`; this is not a new deployment improvement, but a stricter evidence boundary for future data acquisition and calibration.

Stage42-BC turns that data gap into an acquisition plan using official-source checks and local path scans. It ranks UCY Crowd, TrajNet++/AIcrowd, OpenTraj, and ETH/UCY original sources as the high-priority t100 repair path, records DLR AerialMPT and SDD as diagnostic or non-external-repair sources, and deliberately performs no automatic raw download because terms/login/restricted-license review is required. The user-action file is `outputs/stage42_long_research/user_action_required_t100_sources_stage42.md`; Stage42-BC passes `11 / 11` gates and keeps metric/seconds, Stage5C, and SMC claims disabled.

Stage42-BD scans local OpenTraj / ETH / UCY / TrajNet paths for t100-capable source files. It passes `10 / 10` gates with `93` files scanned, `74` parseable, `8` t100-capable, `4` already used, and `4` novel local t100 candidates totaling an estimated `6257` novel t100 windows. This is only inventory evidence; Stage42-BE conversion, no-leakage, and train-only source-CV are still required before any t100 claim can change.

Stage42-BE parses those four novel local t100 candidates for conversion readiness. It passes `12 / 12` gates: all `4` candidates are schema-ready, with `15813` estimated t50 windows and `6257` estimated t100 windows. UCY has enough novel local sources for a source-CV readiness plan after actual conversion; ETH_UCY gains only one small source and remains under-supported. This remains readiness evidence only, not a converted feature store, training run, evaluation run, or t100 success claim.

Stage42-BF then performs actual in-memory schema conversion and causal baseline/source-CV audit for the same four local sources. It passes `12 / 12` gates, with `15058` t50 evaluation windows and `6071` t100 evaluation windows. UCY has positive t100 baseline-family source-CV readiness evidence (`mean holdout improvement vs constant_velocity = 0.607043`, minimum `0.491545`). Stage42-BG then runs a validation-selected protected baseline-family policy source-CV on the converted windows. It passes `13 / 13` gates: UCY local t100 source-CV is positive/easy-safe (`mean improvement = 0.440938`, min `0.438579`, max easy degradation `0.011340`). This still does not permit a global t100 deployment claim because ETH_UCY remains under-supported and TrajNet is not represented in the new local candidates.

Stage42-BH tightens that evidence by deduplicating alternate files from the same scene/source before source-CV. It finds `8` t100-capable files but only `5` independent sources; UCY has `4`, ETH_UCY has `1`, and TrajNet has `0`. Under this stricter protocol UCY still has positive mean t100 gain (`0.483414`) but fails the easy gate (`max easy degradation = 0.063323`), so the honest verdict is partial: UCY t100 needs source-robust easy/harm repair and global t100 remains blocked.

Stage42-BI repairs the UCY independent-source easy-gate failure with a source-robust guard: candidate policies must be positive and easy-safe on every non-holdout source before holdout evaluation. It passes `14 / 14` gates for UCY local t100 support (`mean improvement = 0.445914`, min `0.425313`, max easy degradation `0.011340`, previous BH easy degradation `0.063323`). Global t100 remains blocked because ETH_UCY and TrajNet still lack enough independent t100 sources.

Stage42-BJ packages that post-BI blocker into an explicit acquisition/readiness queue. It passes `14 / 14` gates, preserves the UCY t100 repair, and clarifies strict independent-source deficits: ETH_UCY has only `1` independent t100 source and needs `2` more; TrajNet has `0` and needs `3`. The local inventory is exhausted for these independent-source requirements, no raw/gated data was auto-downloaded, and global t100 remains a raw-frame diagnostic blocker rather than a deployable positive claim.

Stage42-BK then verifies local source/path and loader gaps after BJ. It passes `11 / 11` gates and finds a concrete ETH_UCY repair path: local `ETH-Person/data/*.xml` files provide `5` t100-capable independent conversion candidates (`ETH_UCY` now has `6` local independent t100 groups in the scan, `5` potential new groups vs BJ). These are not counted as converted or evaluated until license/terms confirmation, XML conversion, no-leakage, and train-only source-CV. The same audit explains the TrajNet blocker: `59` local TrajNet files parse as fixed `8/20`-step snippets and provide `0` t100-capable raw-frame sources, so TrajNet still needs longer official/user-provided raw sources.

Stage42-BL runs the ETH-Person XML path as a technical dry-run without upgrading claims. It deduplicates `seq0`/`seq0-interp`, builds `5` strict independent ETH_UCY sources, and completes validation-only source-CV: t100 is safe-positive on all folds with mean improvement `0.683549`, minimum improvement `0.496424`, and maximum easy degradation `-0.014155`. Because local ETH-Person license/terms are still unconfirmed, this remains `fresh_technical_dry_run_terms_unverified`: it is not official converted/evaluated data, not a deployable/global t100 claim, and not metric/seconds-level evidence.

Stage42-BM then audits that license/terms blocker directly. It passes `14 / 14` gates and preserves the positive Stage42-BL technical result, but confirms that the OpenTraj root MIT license is toolkit/software-only for this audit and cannot be treated as ETH-Person dataset permission. No ETH-Person-specific local terms file was found; the official URL is recorded from the OpenTraj README, but terms remain unverified. Therefore ETH-Person XML stays technical-only: no official converted/evaluated claim, no deployable/global t100 claim, no metric/seconds-level claim, no Stage5C, and no SMC.

Stage42-BN adds a stricter source-level time/geometry calibration audit. It identifies source-specific calibration candidates (`ETH_seq_eth`, `ETH_seq_hotel`, `UCY_zara01`, `UCY_zara02`, `UCY_zara03`, `UCY_students03`) with local evidence for parseable H files plus 2.5fps / 0.4s annotation-step timing. It still blocks global M3W metric/seconds claims: SDD remains pixel raw-frame, TrajNet snippets remain dataset-local short snippets, TGSIM remains traffic diagnostic only, and no Stage5C/SMC claim is allowed.

Stage42-BO evaluates those calibrated candidates with source-CV. It finds useful macro signal (`all` +9.05%, `t50` +7.07%, `t100` raw-frame diagnostic +10.41%), but it fails as a deployable policy because one held-out source has severe easy harm (`UCY_students03` easy degradation +103.25%) and another has negative t50 (`ETH_seq_eth` -10.78%). Stage42-BP repairs the easy-harm failure with train+val source/source-family support guards: `easy_degradation_max` becomes 0 and gates pass `11 / 11`. The repaired result is still only limited positive evidence (`all` +5.76%, `t50` +6.19%, `hard/failure` +5.63% macro) because `ETH_seq_eth` t50 remains negative and several sources are fallback-only. This is not a global metric/seconds-level claim, not Stage5C, not SMC, and not a replacement for the protected dataset-local raw-frame wording.

Stage42-BQ then tightens the t50 slice specifically: a t50 switch is allowed only when the same source-family has at least two independent train+val support sources. This removes the remaining negative t50 transfer (`t50_min` becomes 0.0) and keeps `easy_degradation_max` at 0.0, but it also removes all positive t50 folds (`positive_t50_fold_count` is 0). The honest conclusion is t50 non-harm under calibrated-subset support, not positive calibrated t50 transfer.

Stage42-BR audits why positive calibrated-subset t50 disappeared. The answer is mixed: `ETH_seq` has only two calibrated sources, so leave-one-source-out t50 has only one same-family support source and needs one more; `UCY_students` has only one calibrated source and needs two more; `UCY_zara` has enough sources but still has no validation-safe positive t50 policy. Local ETH-Person XML files may repair ETH-style support after terms confirmation, but they remain terms-blocked and are not official/deployable evidence.

Stage42-BS resolves the `UCY_zara` policy/model part of that calibrated t50 blocker without adding new data. It runs a family-only source-CV over `UCY_zara01`, `UCY_zara02`, and `UCY_zara03`, selects candidates and thresholds on validation sources only, and adds a conservative t50 switch-rate guard after the first attempt showed easy harm. The final BS result passes `14 / 14` gates: `rows_total = 51544`, `t50_rows_total = 12750`, t50 macro improvement `+0.247189`, t50 minimum `+0.150958`, hard/failure macro `+0.067158`, easy degradation max `0.012388`, positive t50 folds `3 / 3`. This is source-family-specific annotation-step evidence only: it does not fix `ETH_seq` or `UCY_students`, does not authorize global metric/seconds-level claims, and does not execute Stage5C or SMC.

Stage42-BT checks whether the existing ETH-Person XML technical path can repair the remaining `ETH_seq` calibrated t50 blocker. It cannot. ETH-Person XML h50 has technical positive signal on several ETH-Person holdouts (`technical_h50_mean_improvement_vs_fallback = 0.411217`, `safe_positive_h50_fold_count = 3 / 5`), but the actual `ETH_seq_eth` holdout gets `0.0` improvement under validation-only safety selection. BT passes `13 / 13` gates as an honest blocker confirmation: ETH_seq still needs same-family/source-compatible support, official terms confirmation, or a stronger source-compatible model. ETH-Person terms remain unverified and no official/deployable metric/seconds-level claim is allowed.

## Stage42-A Data Calibration Follow-Up

Stage42 Long Research Mode has started with a fresh data/calibration audit:

- report: `outputs/stage42_long_research/data_calibration_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_a_gate.md`
- user actions: `outputs/stage42_long_research/user_action_required_stage42.md`
- result: Stage42-A gates `7 / 7`

The audit confirms that existing local converted state is sufficient to proceed to Stage42-B external validation and Stage42-C full-waypoint dynamics. It also confirms that global metric and seconds-level claims remain disallowed.

### Stage42-AD Calibration Evidence Refresh

Stage42-AD refreshes the calibration audit by scanning local metadata/README/H.txt/calibration/FPS/scale evidence and separating evidence existence from claim permission:

- report: `outputs/stage42_long_research/calibration_evidence_refresh_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_ad_gate.md`
- user actions: `outputs/stage42_long_research/user_action_required_stage42_calibration.md`
- result: Stage42-AD gates `10 / 10`

Key fresh-run result:

```text
datasets_audited = 7
evidence_files_scanned = 1152
parseable_homography_like = OpenTraj, ETH/UCY, UCY
fps_evidence = SDD, OpenTraj, ETH/UCY, UCY
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
TGSIM = traffic metric diagnostic only
```

Interpretation:

ETH/UCY and UCY have useful local calibration evidence, but this is not enough for a metric or seconds-level pedestrian claim. The allowed claim remains dataset-local raw-frame 2.5D until source-specific homography direction, coordinate convention, annotation stride, frame rate, and scale are verified. Stage5C and SMC remain disabled.

### Stage42-AE Unified Row-Cache Stress Audit

Stage42-AE stress-tests the Stage42-X unified row-level full-waypoint cache instead of only reporting the global mean:

- report: `outputs/stage42_long_research/unified_row_cache_stress_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_ae_gate.md`
- result: Stage42-AE gates `12 / 12`

Key result:

```text
verdict = stage42_ae_unified_row_cache_stress_pass_with_limitations
Stage42-X ADE all = 0.0900
Stage42-X ADE t50 = 0.0611
Stage42-X t50 seed CI low = 0.0537
strong_domains = ETH_UCY, TrajNet, UCY
weak_domain = ETH_UCY for t50/FDE@50 lower bounds
weak_horizon = 25
```

Interpretation:

The global unified row-cache evidence remains strong, but the paper must not claim uniform positivity across every slice. ETH_UCY t50/FDE@50 and horizon=25 should be written as limitations. Claims remain protected dataset-local raw-frame 2.5D.

### Stage42-AF Weak-Slice Validation-Margin Guard Repair

Stage42-AF applies a predeclared validation-margin guard to Stage42-X/Stage42-R row-cache choices:

- report: `outputs/stage42_long_research/weak_slice_guard_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_af_gate.md`
- result: Stage42-AF gates `13 / 13`

Key result:

```text
verdict = stage42_af_weak_slice_guard_repair_pass_with_eth_t50_limitation
guard_threshold = validation score < 0.02
uses_test_metrics_for_threshold = false
horizon25 ADE before = -0.004781
horizon25 ADE after = 0.000000
ADE all = 0.090682
ADE t50 = 0.061094
ADE t50 CI low = 0.053671
ADE hard/failure = 0.094649
easy degradation CI high = 0.006233
ETH_UCY t50 limitation remaining = true
```

Interpretation:

The low-margin guard repairs the horizon=25 negative slice by falling back to the safety floor for low-validation-margin non-UCY domain/horizon choices. It does not use test metrics for threshold selection. This is a safety improvement, not a universal claim: ETH_UCY t50/FDE@50 lower-bound weakness remains a limitation.

### Stage42-AG ETH_UCY T50/FDE Validation-Only Source Repair

Stage42-AG targets the remaining ETH_UCY t50/FDE@50 limitation from Stage42-AF:

- report: `outputs/stage42_long_research/eth_t50_fde_source_repair_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_ag_gate.md`
- result: Stage42-AG gates `13 / 13`

Key result:

```text
verdict = stage42_ag_eth_t50_fde_source_repair_pass
target_slice = ETH_UCY|50
validation_FDE@50_threshold = 0.05
uses_test_metrics_for_threshold = false
ETH_UCY t50 ADE CI low before = -0.013218
ETH_UCY t50 ADE CI low after = 0.002821
ETH_UCY FDE@50 CI low before = -0.041990
ETH_UCY FDE@50 CI low after = 0.021040
ADE all = 0.091656
ADE t50 = 0.064957
ADE t50 CI low = 0.058513
ADE hard/failure = 0.095716
easy degradation CI high = 0.003348
```

Interpretation:

The source repair promotes `stage42j_static_expert` on `ETH_UCY|50` only when validation FDE@50 support is strong and validation ADE@50 is nonnegative; otherwise it falls back to the floor. This repairs the ETH_UCY t50/FDE@50 lower-bound weakness without using test metrics to tune the threshold. Claims remain protected dataset-local raw-frame 2.5D, not metric or seconds-level.

### Stage42-AH Post-Repair Claim Refresh

Stage42-AH refreshes the paper-ready claim matrix after AF/AG:

- report: `outputs/stage42_long_research/post_repair_claim_refresh_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_ah_gate.md`
- result: Stage42-AH gates `11 / 11`

Key result:

```text
verdict = stage42_ah_post_repair_claim_refresh_pass
global ADE all CI low = 0.085258
global ADE t50 CI low = 0.058513
global hard/failure CI low = 0.089767
global easy degradation CI high = 0.003348
global FDE@50 CI low = 0.148230
ETH_UCY t50/FDE limitation = repaired
horizon25 = floor/non-harm, not positive dynamics
TrajNet|100 = safety-limited
metric/seconds claim = rejected
```

Interpretation:

The post-repair claim is stronger than Stage42-AE: the old horizon=25 negative slice is now floor/non-harm and ETH_UCY t50/FDE@50 lower bounds are positive. It is still bounded evidence. Horizon=25 should not be described as a positive dynamics contribution, t100 remains raw-frame diagnostic with a TrajNet|100 safety limit, and metric/seconds/true-3D/foundation claims remain disallowed.

### Stage42-AI TrajNet T100 Easy-Safety Repair

Stage42-AI targets the TrajNet|100 safety limitation recorded by Stage42-AH:

- report: `outputs/stage42_long_research/trajnet_t100_safety_repair_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_ai_gate.md`
- result: Stage42-AI gates `13 / 13`

Key result:

```text
verdict = stage42_ai_trajnet_t100_safety_repair_pass
target_slice = TrajNet|100
validation_easy_nonharm_threshold = 0.0
uses_test_metrics_for_threshold = false
TrajNet|100 ADE CI low after = 0.048714
TrajNet|100 easy CI high before = 0.084984
TrajNet|100 easy CI high after = 0.000000
global ADE all CI low = 0.085978
global ADE t50 CI low = 0.058513
global ADE t100 raw-frame diagnostic CI low = 0.068349
global hard/failure CI low = 0.090662
global easy degradation CI high = 0.001168
```

Interpretation:

The repair uses validation easy-degradation only. It selects the easy-safe positive validation source for `TrajNet|100`, otherwise floor. This removes the TrajNet|100 easy harm while preserving positive raw-frame diagnostic t100 evidence. It still does not allow seconds-level, metric, true-3D, foundation, Stage5C, or SMC claims.

### Stage42-AJ Post-Repair Paper Package Refresh

Stage42-AJ refreshes the paper package with AD-AI evidence:

- report: `outputs/stage42_long_research/paper_package_post_repair_refresh_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_aj_gate.md`
- result: Stage42-AJ gates `10 / 10`

Key result:

```text
verdict = stage42_aj_post_repair_paper_package_refresh_pass
paper_files_refreshed = 9 / 9
included = calibration refresh, horizon25 repair, ETH_UCY t50/FDE repair, post-repair claim matrix, TrajNet t100 safety repair
metric/seconds claim = rejected
t100 seconds claim = rejected
Stage5C = false
SMC = false
```

Interpretation:

The paper package is now current through Stage42-AI. The package can claim protected dataset-local raw-frame 2.5D full-waypoint evidence with repaired weak slices. It still must reject metric, seconds-level, true-3D, foundation, Stage5C, SMC, and ungated-neural deployment claims.

## Stage42-B External Validation Follow-Up

Stage42-B rebuilt a source-level/fold stress protocol over the frozen external evaluation pool and reran the protected package comparisons:

- report: `outputs/stage42_long_research/external_validation_stage42.md`
- source split: `outputs/stage42_long_research/external_source_split_stage42.json`
- gate: `outputs/stage42_long_research/stage42_stage_b_gate.md`
- result: Stage42-B gates `10 / 10`

Key fresh-run result:

```text
frozen_eval_pool_rows = 66303
evaluated_rows = 55528
protected_M3W_all_ADE_improvement = 0.2103
protected_M3W_t50_ADE_improvement = 0.1365
protected_M3W_t100_raw_frame_diagnostic_ADE_improvement = 0.1469
protected_M3W_hard_failure_ADE_improvement = 0.2038
protected_M3W_easy_degradation = -0.1451
ungated_neural_all_ADE_improvement = 0.2966
ungated_neural_easy_degradation = 1.2459
verdict = stage42_b_external_validation_pass_protected_neural_not_ungated
```

This confirms the protected neural candidate under the Stage37/teacher floor, and it also confirms the safety failure of ungated neural endpoint dynamics. Negative easy degradation means no easy-case harm under the report's metric convention. The result remains dataset-local raw-frame 2.5D evidence only; it is not metric, seconds-level, true 3D, Stage5C, or SMC.

## Stage42-C Full-Waypoint Dynamics Follow-Up

Stage42-C evaluates actual reconstructed future waypoint labels rather than only endpoint FDE:

- report: `outputs/stage42_long_research/full_waypoint_dynamics_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_c_gate.md`
- result: Stage42-C gates `12 / 12`

Key fresh-run result:

```text
full_waypoint_sequence_model = full_trajectory_ensemble
positive_full_waypoint_domains = ETH_UCY, TrajNet
protected_full_waypoint_ADE_all = 0.1858
protected_full_waypoint_ADE_t50 = 0.1480
protected_full_waypoint_ADE_t100_raw_frame_diagnostic = 0.2286
protected_full_waypoint_ADE_hard_failure = 0.1952
protected_full_waypoint_easy_degradation = 0.0000
protected_full_waypoint_FDE_all = 0.1938
protected_full_waypoint_FDE_t50 = 0.2158
protected_full_waypoint_near_collision_delta_005 = 0.0086
ungated_full_waypoint_easy_degradation = 1.2459
```

Interpretation:

The protected full-waypoint sequence model strengthens the world-state claim because it is evaluated on reconstructed future waypoint labels and is positive on two external domains. It is not a complete replacement for the composite-tail linear bridge yet: composite-tail has higher all-ADE, while the full-waypoint sequence model is stronger on t+50/t+100 raw-frame waypoint metrics. Ungated full-waypoint neural remains unsafe, so the Stage37/teacher floor and safe switch stay in the deployable path.

## Stage42-D/E Causal Ablation And Safety Floor Follow-Up

Stage42-D adds a causal ablation evidence audit:

- report: `outputs/stage42_long_research/causal_ablation_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_d_gate.md`
- result: Stage42-D gates `12 / 12`
- boundary: not every component was retrained inside Stage42-D; fresh rows cover safety/floor/full-waypoint ablations, while historical no-history/no-neighbor/no-scene-goal/no-interaction/no-JEPA/no-Transformer/no-fallback evidence is cached-verified.

Stage42-E studies whether the Stage37/teacher safety floor can be removed:

- report: `outputs/stage42_long_research/safety_floor_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_e_gate.md`
- result: Stage42-E gates `12 / 12`

Key fresh-run result:

```text
best_policy_family = current_composite_tail_policy
best_policy_source = cached_verified_policy_fresh_eval
best_all = 0.2102513255185352
best_t50 = 0.13652231450154184
best_t100_raw_frame_diagnostic = 0.14694086716388166
best_hard_failure = 0.20384916307933942
best_easy_degradation = 0.0
floor_necessity_conclusion = teacher_floor_required_for_current_deployment
ungated_endpoint_easy_degradation = 1.2458611044726973
```

Interpretation:

The Stage37/teacher floor remains necessary for current deployment. Ungated neural has stronger raw all/t50/hard numbers but fails safety with easy degradation around `1.2459` and worse proximity/collision. Internal self-gate, uncertainty gate, harm gate, and conformal gate show large raw lift but exceed the collision safety ceiling in this fresh study. Teacher-repaired and composite-tail protected policies remain the deployable path. This is still dataset-local raw-frame 2.5D evidence, not metric, seconds-level, true 3D, Stage5C, or SMC.

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

## Stage42-U UCY Candidate Bridge Audit

```text
source = fresh_run
report = outputs/stage42_long_research/ucy_candidate_bridge_stage42.md
gate = outputs/stage42_long_research/stage42_stage_u_gate.md
verdict = stage42_u_ucy_endpoint_to_full_bridge_failed_blocker
gates = 7 / 8
```

Stage42-U answers a narrow but important question after Stage42-T: can the strict Stage41 pure-UCY endpoint neural candidate become the missing non-floor UCY source for Stage42 full-waypoint evaluation?

The answer is no under the tested linear endpoint-to-full bridge. The Stage41 pure-UCY endpoint candidate is available and row-aligned with Stage42 val/test rows, but when its endpoint residual is linearly interpolated into full waypoints, validation and UCY test full-waypoint metrics are negative:

```text
UCY_zara03_test ADE all = -0.070821
UCY_zara03_test ADE t50 = -0.492070
UCY_zara03_test hard/failure = -0.083302
UCY_zara03_test easy degradation = 0.566646
```

This is a blocker diagnosis, not a success. It proves that endpoint-FDE success cannot be counted as full-waypoint world-state success. The next aligned action is to train/cache a UCY-aware full-waypoint candidate source or learn a validation-selected waypoint-shape bridge. Stage5C and SMC remain disabled, and no metric/seconds-level claim is made.

## Stage42-V Strict Pure-UCY Full-Waypoint Candidate

```text
source = fresh_run
report = outputs/stage42_long_research/ucy_full_waypoint_candidate_stage42.md
gate = outputs/stage42_long_research/stage42_stage_v_gate.md
verdict = stage42_v_ucy_full_waypoint_candidate_pass
gates = 11 / 11
```

Stage42-V directly repairs the Stage42-U blocker by training a UCY-aware full-waypoint model instead of linearly interpolating an endpoint residual. The protocol is strict and source-heldout: train on UCY `students01`/`students03`, validate on UCY `zara01`, and test once on UCY `zara02`/`zara03`.

Best 3-seed result:

```text
best_trial = ucy_full_waypoint_t50_hard
ADE all = 0.220755
ADE t50 = 0.290332
ADE t50 CI low = 0.231725
ADE t100 raw-frame diagnostic = 0.147461
hard/failure = 0.229484
easy degradation = 0.000000
FDE t50 = 0.334459
```

This is a meaningful UCY full-waypoint candidate source. It still remains dataset-local/raw-frame 2.5D evidence, not metric/seconds-level, not true 3D, and not Stage5C/SMC. The next step is to integrate this UCY source into the Stage42-R/S row-combo policy rather than treating it as a standalone final model.

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

Stage42-I is not a full pass. The full static+sequence model is ADE-negative, but the no-static-context sequence model is positive on all/t50/hard and keeps easy degradation at zero. This points to static/context gating as the next repair, not a claim that the current full sequence-to-waypoint head is deployable.

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

Stage42-K has completed as a fresh-run checkpoint experiment:

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
```

It exists because Stage42-J showed that static/context is useful only when gated, but Stage42-J itself was a cached-checkpoint expert gate rather than a new trained checkpoint. Stage42-K shows that the learned static gate/dropout rule can be trained into a fresh checkpoint and improve over the failed Stage42-I full static+sequence head while preserving easy cases.

It is not the new best deployable full-waypoint policy: Stage42-J remains stronger on ADE all/t50/hard and FDE t50, while Stage42-K still has negative ADE t50. The next repair should make the learned static gate horizon-aware, especially for t+50.

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

It repairs Stage42-K's t+50 ADE sign and improves the fresh checkpoint on all/hard/FDE t50 without easy degradation. It still does not surpass the Stage42-J policy-level static gate, so the deployable full-waypoint static-gated path remains Stage42-J unless a later fresh checkpoint catches up.

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

It is partial, not a pass: FDE t50 improves over Stage42-L, but ADE t50 remains negative and ADE all/hard are weaker than Stage42-L. The teacher signal is too coarse because it distills domain/horizon expert alpha rather than row-level gain/harm. Stage42-L remains the best fresh checkpoint; Stage42-J remains the strongest static-gated full-waypoint evidence overall.

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

It is not a t+50 repair. ADE all (`0.0250`) and hard/failure (`0.0269`) improve over Stage42-L/M with easy degradation `0.0`, but ADE t50 is negative (`-0.0278`). The diagnosis is now sharper: row-level alpha supervision alone still does not teach the model which t+50 rows are safe to switch. Next work should train an explicit gain/harm/switchability selector head or a t+50-specific teacher ensemble.

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

After fixing normalization to use train-split statistics only, Stage42-O is a useful partial result rather than a t+50 pass. It improves all and hard/failure over Stage42-N and keeps easy degradation below the mean 2% gate, but ADE t50 remains slightly negative.

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

Interpretation: Stage42-P repairs the mean ADE t50 sign from Stage42-O (`-0.0008` to `+0.0066`) while keeping all/hard positive and easy degradation below 2%. It is still not a paper-stable t50 claim because the 3-seed t50 CI low is negative; next work should add bootstrap/seeds and combine it with the Stage42-J static expert policy.

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

Verification: Stage42-Y runner passed, focused Stage42-Y pytest passed with 3 tests, and the full repository test suite passed with 327 tests.

## Stage42-Z Paper Claim Evidence Audit

```text
source = fresh_audit_from_stage42_wxy_and_paper_package_artifacts
verdict = stage42_z_paper_claim_evidence_audit_pass
gates = 16 / 16
paper_ready_scope = protected_2p5d_raw_frame_world_state_candidate
not_ready_scope = true_3d_metric_seconds_foundation_or_stage5c_smc
stage5c_executed = false
smc_enabled = false
```

Stage42-Z makes the claim boundary explicit for the paper package: unified row-level full-waypoint evidence, t50 positivity, UCY source contribution, history-token contribution, protected external floor, and protected full-waypoint dynamics are supported. Ungated neural replacement, metric/seconds-level claims, true-3D/foundation claims, and uniform goal/scene or neighbor/interaction positivity are not supported as main claims.

## Stage42-AA Retrained Ablation Matrix

```text
source = fresh_matrix_from_stage42g_rerun_plus_stage42h_i_d_z
verdict = stage42_aa_retrained_ablation_matrix_pass_with_jepa_transformer_boundary
gates = 15 / 15
fresh_required_coverage = 11 / 12
stage5c_executed = false
smc_enabled = false
```

Stage42-AA reruns the Stage42-G retrained ablation and unifies the required ablation evidence. It shows 11 of 12 requested ablation categories have fresh Stage42 evidence; no-JEPA remains cached negative architecture evidence and is not relabeled as fresh retraining. Teacher-floor removal is unsafe, so the Stage37/teacher safety floor remains required for deployment.

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

## Stage42-AX Repaired Protocol Robustness

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
stage5c_executed = false
smc_enabled = false
```

Stage42-AX verifies the Stage42-AW repaired validation-support protocol without reusing test metrics for threshold selection. The repaired protocol supports global positive bootstrap evidence and positive TrajNet/UCY source-level evidence. It also keeps the remaining limitation explicit: horizon 100 remains raw-frame diagnostic and has an easy-safety weak slice, so uniform horizon success and metric/seconds-level claims remain disallowed.

## Stage42-AY AW T100 Easy-Safety Repair

```text
source = fresh_run
verdict = stage42_ay_t100_easy_safety_repair_pass
gates = 17 / 17
guarded_slice = TrajNet|100
h100_easy_before = 0.023961
h100_easy_after = -0.006504
h100_easy_CI_high = 0.009833
global_all_after = 0.305467
global_t50_after = 0.289698
global_t100_raw_frame_diagnostic_after = 0.067836
global_hard_failure_after = 0.279764
stage5c_executed = false
smc_enabled = false
```

Stage42-AY applies a stricter validation-only t100 easy-safety guard to the Stage42-AW repaired protocol. `TrajNet|100` is guarded back to the floor because its validation easy degradation is above the strict non-harm threshold, while `UCY|100` remains active. This repairs the Stage42-AX h100 easy-safety weak slice but reduces t100 diagnostic gain; it remains dataset-local raw-frame evidence and needs future held-out confirmation for stronger paper claims.

## Stage42-AZ AY Shadow-Holdout T100 Robustness Audit

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
stage5c_executed = false
smc_enabled = false
```

Stage42-AZ tests the Stage42-AY strict t100 guard on a source-level shadow split built only from original train sources. It finds that the AY strict guard is not independently robust for t100 easy safety: ETH_UCY t100 easy harm appears on the shadow holdout. A more conservative source-support guard keeps all/t50/hard positive and protects easy cases, but it gives up positive t100 gain on this shadow holdout. This is a useful safety result and a claim-boundary result, not a new t100 success: t100 remains raw-frame diagnostic and should not be written as seconds-level or uniformly robust long-horizon world-model evidence.

## Stage42-BA Train-Only T100 Source-CV Repair

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
stage5c_executed = false
smc_enabled = false
```

Stage42-BA converts the Stage42-AZ t100 warning into an explicit train-only source-CV guard. It leaves one original-train t100-capable source out at a time, selects policy support without final test metrics, and only allows a domain t100 slice if it has at least two safe-positive source folds. No domain currently satisfies that rule. The repaired deployment boundary is therefore clear: all/t50/hard remain positive and easy-safe under the guard, but t100 positive gain is not supported and is guarded to the causal floor.

## Stage42-BU UCY_students T50 Source-Support Audit

```text
source = fresh_ucy_students_t50_source_support
verdict = stage42_bu_ucy_students_t50_source_support_pass_blocker_narrowed
gates = 14 / 14
independent_t50_capable_sources = UCY_students01, UCY_students03
additional_independent_t50_sources_still_needed = 1
source_cv_ready = false
ucy_students_t50_support_repaired = false
stage5c_executed = false
smc_enabled = false
```

Stage42-BU narrows the remaining calibrated t50 blocker for `UCY_students`. The local `students001` source is t50-capable and independent from the existing `students003` calibrated source, but `students002` is too short for t50 and duplicate/alternate `students001/003` files are not counted as independent. UCY_students therefore still needs one more independent t50-capable same-family source before a protected train/val/holdout source-CV t50 repair can be attempted.

## Stage42-BV Source Acquisition / Blocker Matrix

```text
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
stage5c_executed = false
smc_enabled = false
```

Stage42-BV turns the remaining source-support and claim-boundary blockers into an explicit acquisition/status matrix. It preserves the positive Stage37/M3W-Neural protected evidence, but refuses to upgrade any terms-blocked, snippet-only, or source-under-supported result into a deployable global claim. The active blockers are:

- `ETH_seq_t50_source_support`: ETH-Person XML has technical h50 signal, but terms remain unverified and the dry-run did not safely repair the actual `ETH_seq_eth` holdout.
- `UCY_students_t50_source_support`: BU found `students001` as a new t50-capable independent source, but one more independent students-family source is still needed.
- `TrajNet_raw_long_t100_source_support`: local TrajNet files are short challenge snippets, not raw long tracks for raw-frame t100 source-CV.
- `ETH_UCY_global_t100_source_support`: ETH-Person XML t100 dry-run remains technical-only until terms and official conversion are resolved.
- `global_metric_seconds_claim`: only source-specific calibration evidence exists; global M3W remains raw-frame / dataset-local.

No Stage5C execution, no SMC, no metric/seconds-level overclaim, and no automatic download occurred.

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
floor_free_neural_deployable = false
teacher_floor_context_required = true
stage5c_executed = false
smc_enabled = false
```

Stage42-BY rechecks the slice-level t50 blocker from Stage42-BX with the Stage42-AW train-only internal validation policy. Both `TrajNet|50` and `UCY|50` become protected-positive slices, while the claim boundary stays conservative: the result depends on the teacher/floor rollout context and protected fallback, so it is not floor-free neural world dynamics.

## Stage42-BZ Protected T50 Repair Statistical Evidence

```text
source = fresh_stage42_bz_t50_repair_statistical_evidence
verdict = stage42_bz_t50_repair_statistical_evidence_pass
gates = 13 / 13
bootstrap_n = 3000
robust_t50_slices = TrajNet|50, UCY|50
target_union_t50_CI = [28.52%, 29.45%]
target_union_easy_degradation_CI_high = -25.16%
TrajNet|50 t50_CI = [29.80%, 30.67%]
UCY|50 t50_CI = [23.02%, 26.08%]
floor_free_neural_deployable = false
stage5c_executed = false
smc_enabled = false
```

Stage42-BZ adds 3000-sample bootstrap evidence to the Stage42-BY protected repair. Both repaired t50 slices are statistically positive with easy-safety preserved. The deployment boundary is unchanged: Stage37/teacher floor and protected safe-switch remain required.

## Stage42-CA Post-BZ Paper Package Refresh

```text
source = fresh_synthesis_from_stage42_by_bz_artifacts
verdict = stage42_ca_post_bz_paper_package_refresh_pass
gates = 10 / 10
paper_files_refreshed = 9 / 9
target_union_t50_CI = [28.52%, 29.45%]
target_union_easy_degradation_CI_high = -25.16%
floor_free_neural_deployable = false
stage5c_executed = false
smc_enabled = false
```

Stage42-CA makes the BY/BZ evidence paper-package-visible across the Stage42 outline, method draft, experiment/ablation tables, failure taxonomy, model/data cards, reproducibility note, and A-journal gap analysis. It is a paper-evidence consistency refresh, not new model training.

## Stage42-CB Protected T50 Source Robustness Audit

```text
source = fresh_stage42_cb_t50_source_robustness_audit
verdict = stage42_cb_t50_source_robustness_pass_with_source_diversity_limit
gates = 11 / 11
robust_major_source_slices = TrajNet|50, UCY|50
concentration_limited_slices = TrajNet|50, UCY|50
broad_source_generalization_claim_allowed = false
TrajNet|50 largest_source_fraction = 99.08%
UCY|50 largest_source_fraction = 100.00%
```

Stage42-CB adds the source-level caveat to the t50 repair evidence. The available major sources are robust-positive, but source diversity is limited; broad source-level generalization is not yet allowed.

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

Stage42-CC confirms the Stage42-CB caveat rather than repairing it. The local scan found t50-capable files, but none can be counted as an unused independent ready-to-claim real external source. Four are alternate/current-source representations and one is synthetic/diagnostic. The M3W-Neural v1 claim therefore remains protected dataset-local/raw-frame 2.5D with source-diversity limitation; broad source-level generalization still requires legally enabled independent top-down pedestrian data plus conversion, no-leakage, validation-only selection, and final test.

## Stage42-CD Source Diversity Acquisition Package

```text
source = fresh_stage42_cd_source_diversity_acquisition_package
verdict = stage42_cd_source_diversity_acquisition_package_pass
gates = 13 / 13
official_targets = 5
auto_download_targets = 0
converted_datasets_now = 0
source_diversity_repair_ready_now = false
broad_source_generalization_claim_allowed = false
stage5c_executed = false
smc_enabled = false
```

Stage42-CD adds the official/manual acquisition package for the remaining source-diversity blocker. The target list covers UCY Crowd Data, ETH/BIWI, TrajNet++, OpenTraj as a toolkit/reference, and an additional legal top-down pedestrian/drone source. It intentionally does not download, convert, or claim success. Current M3W-Neural v1 remains protected dataset-local/raw-frame 2.5D; broad source-level generalization still waits on legal independent source conversion and source-CV.

## Stage42-CE Source Diversity Conversion Preflight

```text
source = fresh_stage42_ce_source_diversity_conversion_preflight
verdict = stage42_ce_source_diversity_conversion_preflight_pass
gates = 12 / 12
targets_with_local_path = 4
targets_with_schema_possible = 4
targets_with_t50_files = 3
targets_with_t100_files = 3
targets_with_independent_t50_candidates = 0
targets_source_cv_ready_now = 0
converted_datasets_now = 0
evaluated_datasets_now = 0
```

Stage42-CE confirms that local paths are present and parseable for several source-diversity targets, but no target is ready for a source-CV repair claim. This is useful engineering evidence for future conversion, not a model improvement or generalization claim. The current deployable model and claim boundary are unchanged.

## Stage42-CF Source Conversion Legal Gate

```text
source = fresh_stage42_cf_source_conversion_legal_gate
verdict = stage42_cf_source_conversion_legal_gate_pass
gates = 13 / 13
source_cv_ready_now = 0
conversion_allowed_now_count = 0
converted_datasets_now = 0
evaluated_datasets_now = 0
```

Stage42-CF makes the source-diversity blocker enforceable. Local parseability from Stage42-CE is now passed through a legal/source-identity gate that refuses conversion until explicit official terms confirmation and an independent source-CV-ready held-out source exist. It writes a terms confirmation template, but the template is not permission. M3W-Neural v1 remains protected dataset-local/raw-frame 2.5D; broad source-level generalization is still not claimed.

## Stage42-CG Source Terms Confirmation Validator

```text
source = fresh_stage42_cg_source_terms_confirmation_validator
verdict = stage42_cg_source_terms_confirmation_validator_pass
gates = 11 / 11
terms_accepted_targets = 0
conversion_ready_targets = 0
converted_datasets_now = 0
evaluated_datasets_now = 0
```

Stage42-CG validates the CF terms confirmation template and writes a conversion readiness manifest. The current template is intentionally blank, so every source remains blocked. This preserves the M3W-Neural v1 evidence boundary: no legal/source-diversity conversion has happened yet, and broad source-level generalization still requires explicit terms confirmation plus a later no-leakage/source-CV/final-test conversion stage.

## Stage42-CH Metric/Time Claim Guard

```text
source = fresh_stage42_ch_metric_time_claim_guard
verdict = stage42_ch_metric_time_claim_guard_pass
gates = 11 / 11
source_specific_metric_time_candidates = 6
conversion_ready_targets = 0
global_metric_claim_allowed = false
global_seconds_claim_allowed = false
restricted_subset_metric_seconds_claim_allowed_now = false
```

Stage42-CH keeps the calibration story honest. ETH/UCY source-specific metric/time candidates exist, but they are not paper-allowed metric/seconds claims because legal conversion readiness is still zero and no restricted-subset final evaluation has run. The deployable M3W-Neural v1 claim therefore remains protected dataset-local/raw-frame 2.5D, not metric, not seconds-level, not true 3D, and not foundation-scale.

## Stage42-Z Post-CH Paper Claim Evidence Audit

```text
source = fresh_audit_from_stage42_wxy_and_paper_package_artifacts
verdict = stage42_z_paper_claim_evidence_audit_pass
gates = 22 / 22
claim_rows = 13
```

The paper claim matrix now includes the CG legal/source terms validator and the CH metric/time claim guard. This keeps M3W-Neural v1 paper-ready only as a protected dataset-local/raw-frame 2.5D world-state candidate. Legal source-diversity conversion is not ready, and restricted metric/seconds subset claims remain blocked even though source-specific calibration candidates exist.

## Stage42-CI Context Contribution Forensics

```text
source = fresh_synthesis_from_stage42_ablation_and_claim_audits
verdict = stage42_ci_context_contribution_forensics_pass
gates = 13 / 13
```

The context contribution map is now explicit: baseline-family rollout context is the dominant mechanism, causal history tokens are a supported core component, and domain expert is a smaller guarded component. Goal/scene and neighbor/interaction context remain mixed under the current protocols, while JEPA and Transformer are not independent deployable main claims. This sharpens the model-card language for M3W-Neural v1: the protected safety-floor mechanism is part of the contribution, not just an implementation detail.

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
- artifacts audited: `28`.
- artifacts with passing gates: `28`.
- source-label counts: `{'fresh_run': 27, 'cached_verified': 1}`.
- worktree caveat artifacts recorded: `0`.
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
- gate: `27 / 27`; verdict `stage42_dm_reviewer_replay_package_pass`.
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

<!-- STAGE42_FQ_H100_SOURCE_SUPPORT_REPAIR_QUEUE:START -->
## Stage42-FQ H100 Source-Support Repair Queue

- source: `fresh_stage42_h100_source_support_repair_queue`
- role: local source-support repair queue for FP h100 blockers; no conversion, no training, no auto-download.
- gate: `15 / 15`; verdict `stage42_fq_h100_source_support_repair_queue_pass`.
- weak keys: `['TrajNet|100', 'UCY|100']`.
- local gap summary: `{'ETH_UCY': {'files': 18, 't100_files': 7, 'independent_t100_groups': 6, 'short_or_non_t100_files': 11}, 'TrajNet': {'files': 59, 't100_files': 0, 'independent_t100_groups': 0, 'short_or_non_t100_files': 59}, 'UCY': {'files': 24, 't100_files': 6, 'independent_t100_groups': 4, 'short_or_non_t100_files': 18}}`.
- TrajNet|100 status: no local long raw h100 TrajNet source; user must provide or confirm official longer source.
- UCY|100 status: local UCY h100 candidates exist but are terms-unverified and require conversion/no-leakage/source-CV before use.
- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.
- verification: `{'runner': '.venv-pytorch/bin/python run_stage42_h100_source_support_repair_queue.py -> 15/15', 'focused_pytest': '.venv-pytorch/bin/python -m pytest tests/test_stage42_h100_source_support_repair_queue.py -> 4 passed', 'full_pytest': '.venv-pytorch/bin/python -m pytest tests -> 836 passed'}`.
<!-- STAGE42_FQ_H100_SOURCE_SUPPORT_REPAIR_QUEUE:END -->

<!-- STAGE42_FR_UCY_H100_TERMS_GATED_PREFLIGHT:START -->
## Stage42-FR UCY H100 Terms-Gated Conversion Preflight

- source: `fresh_stage42_ucy_h100_terms_gated_conversion_preflight`
- role: file-level UCY h100 candidate preflight from FQ; no conversion, no training, no auto-download.
- gate: `14 / 14`; verdict `stage42_fr_ucy_h100_terms_gated_preflight_pass`.
- candidates: `6` total, `2` target-family candidates.
- conversion_preflight_ready_count: `0`; blockers `['terms_not_accepted', 'terms_acceptance_date_missing', 'allowed_use_missing', 'redistribution_policy_unknown', 'derived_data_policy_unknown', 'local_path_confirmation_missing', 'source_identity_missing', 'confirmed_by_user_missing']`.
- recommended first sources after user confirmation: `['UCY_zara02', 'UCY_zara01']`.
- Boundary: protected source-level raw-frame 2.5D; no converted dataset claim, no metric/seconds claim, no true 3D, no Stage5C, no SMC.
- verification: `{'runner': '.venv-pytorch/bin/python run_stage42_ucy_h100_terms_gated_conversion_preflight.py -> 14/14', 'focused_pytest': '.venv-pytorch/bin/python -m pytest tests/test_stage42_ucy_h100_terms_gated_conversion_preflight.py -> 4 passed', 'full_pytest': '.venv-pytorch/bin/python -m pytest tests -> 840 passed'}`.
<!-- STAGE42_FR_UCY_H100_TERMS_GATED_PREFLIGHT:END -->

<!-- STAGE42_FS_UCY_H100_TERMS_INTAKE_VALIDATOR:START -->
## Stage42-FS UCY H100 Terms Intake Validator

- source: `fresh_stage42_ucy_h100_terms_intake_validator`
- role: validates candidate-level UCY h100 terms intake and writes a guarded conversion queue; no conversion, training, download, or evaluation.
- gate: `14 / 14`; verdict `stage42_fs_ucy_h100_terms_intake_validator_pass`.
- candidate_rows_validated: `6`; target_family_candidates `2`.
- terms_ready_candidates: `0`; guarded_conversion_queue_count `0`.
- top blockers: `{'allowed_use_missing': 6, 'confirmed_by_user_missing': 6, 'derived_data_policy_unknown': 6, 'local_path_confirmation_missing': 6, 'redistribution_policy_unknown': 6, 'source_identity_missing': 6, 'terms_acceptance_date_missing': 6, 'terms_not_accepted': 6}`.
- Boundary: protected source-level raw-frame 2.5D; no converted dataset claim, no metric/seconds claim, no true 3D, no Stage5C, no SMC.
- verification commands: `{'runner': '.venv-pytorch/bin/python run_stage42_ucy_h100_terms_intake_validator.py -> 14/14', 'focused_pytest': '.venv-pytorch/bin/python -m pytest tests/test_stage42_ucy_h100_terms_intake_validator.py -> 4 passed', 'full_pytest': '.venv-pytorch/bin/python -m pytest tests -> 844 passed'}`.
<!-- STAGE42_FS_UCY_H100_TERMS_INTAKE_VALIDATOR:END -->

<!-- STAGE42_FT_UNIFIED_GUARDED_CONVERSION_QUEUE:START -->
## Stage42-FT Unified Guarded Conversion Queue

- source: `fresh_stage42_unified_guarded_conversion_queue`
- role: unifies global source readiness and UCY H100 candidate readiness into one non-executing guarded conversion queue.
- gate: `12 / 12`; verdict `stage42_ft_unified_guarded_conversion_queue_pass`.
- source_ready_targets: `0`; h100_ready_candidates `0`; unified_queue_count `0`.
- blocked_action_count: `11`; downloaded/converted/evaluated now `0` / `0` / `0`.
- Boundary: queue only; no converted dataset claim, no metric/seconds claim, no true 3D, no Stage5C, no SMC.
- verification commands: `{'runner': '.venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py -> 12/12', 'focused_pytest': '.venv-pytorch/bin/python -m pytest tests/test_stage42_unified_guarded_conversion_queue.py -> 4 passed', 'full_pytest': '.venv-pytorch/bin/python -m pytest tests -> 848 passed'}`.
<!-- STAGE42_FT_UNIFIED_GUARDED_CONVERSION_QUEUE:END -->

<!-- STAGE42_FU_MODULE_CONTRIBUTION_LEDGER:START -->
## Stage42-FU Module Contribution Ledger

- source: `fresh_stage42_module_contribution_ledger_from_aa_y_bw_ec_dp_de`
- role: machine-readable claim ledger over AA/Y/BW/EC/DP/DE evidence; no new training or threshold tuning.
- gate: `14 / 14`; verdict `stage42_fu_module_contribution_ledger_pass`.
- main claim modules: `['history', 'domain_expert', 'safe_switch', 'teacher_floor', 'group_consistency_full_waypoint', 'full_waypoint_shape', 'endpoint_bridge']`.
- blocked/auxiliary modules: `['scene_goal', 'neighbor_interaction', 'JEPA', 'Transformer']`.
- Core supported claims: history, domain expert, safe-switch/teacher floor, and source-level group-consistency full-waypoint.
- Blocked as main independent claims under current evidence: JEPA downstream lift, Transformer-only contribution, scene/goal, neighbor/interaction, ungated neural/global metric/seconds.
- verification commands: `{'runner': '.venv-pytorch/bin/python run_stage42_module_contribution_ledger.py -> 14/14', 'focused_pytest': '.venv-pytorch/bin/python -m pytest tests/test_stage42_module_contribution_ledger.py -> 4 passed', 'full_pytest': '.venv-pytorch/bin/python -m pytest tests -> 852 passed'}`.
<!-- STAGE42_FU_MODULE_CONTRIBUTION_LEDGER:END -->

<!-- STAGE42_FV_CLAIM_BOUNDARY_LINTER:START -->
## Stage42-FV Claim Boundary / No-Overclaim Linter

- source: `fresh_stage42_claim_boundary_linter_from_paper_package_and_fu`
- gate: `15 / 15`; verdict `stage42_fv_claim_boundary_linter_pass`.
- scanned files: `15`; violations: `0`.
- role: paper-package claim hygiene guard; no training, no threshold tuning, no conversion.
- boundary: M3W remains protected dataset-local/raw-frame 2.5D; no true 3D/foundation/global metric/seconds/Stage5C/SMC claim.
- blocked as independent main claims: JEPA, Transformer, scene/goal, neighbor/interaction.
- verification commands: `{'runner': '.venv-pytorch/bin/python run_stage42_claim_boundary_linter.py -> 15/15', 'focused_pytest': '.venv-pytorch/bin/python -m pytest tests/test_stage42_claim_boundary_linter.py tests/test_stage42_module_contribution_ledger.py -> 9 passed', 'full_pytest': '.venv-pytorch/bin/python -m pytest tests -> 857 passed'}`.
<!-- STAGE42_FV_CLAIM_BOUNDARY_LINTER:END -->

<!-- STAGE42_FW_SOURCE_ACTION_CONSOLIDATOR:START -->
## Stage42-FW Source Action Consolidator

- source: `fresh_stage42_source_action_consolidator_from_existing_blockers`
- gate: `16 / 16`; verdict `stage42_fw_source_action_consolidator_pass`
- consolidated actions: `10`; categories `{'legal_terms_and_local_path': 5, 'h100_weak_horizon_source_support': 2, 'domain_closure': 3}`
- top actions: `['FW-TERMS-ucy_crowd_original', 'FW-H100-TrajNet|100', 'FW-DOMAIN-TrajNet', 'FW-DOMAIN-UCY', 'FW-H100-UCY|100']`
- conversion_ready_now: `0`; blocked_action_count: `11`
- This is a source/legal/horizon action router only: no download, conversion, training, evaluation, metric/seconds claim, Stage5C execution, or SMC.
- Highest-value path remains UCY terms/path confirmation plus guarded conversion/no-leakage/source-CV; TrajNet h100 needs a longer legal source because local snippets are too short.
- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; not true 3D, not foundation, not metric/seconds-level.
<!-- STAGE42_FW_SOURCE_ACTION_CONSOLIDATOR:END -->

<!-- STAGE42_FX_OBJECTIVE_COVERAGE_AUDIT:START -->
## Stage42-FX Objective Coverage Audit

- source: `fresh_stage42_objective_coverage_audit_from_current_evidence`
- gate: `15 / 15`; verdict `stage42_fx_objective_coverage_audit_pass`.
- objectives covered: `6`; blocked objectives `['A']`; partial objectives `['B', 'C', 'D']`; passed objectives `['E']`.
- current best status: `protected_dataset_local_raw_frame_2_5d_candidate`.
- highest-priority next action: `FW-TERMS-ucy_crowd_original`.
- role: requirement coverage audit for the active Stage42 A-F long objective; no training, no download, no conversion, no threshold tuning.
- boundary: goal remains active and incomplete; M3W remains protected dataset-local/raw-frame 2.5D; no true 3D/foundation/global metric/seconds/Stage5C/SMC claim.
<!-- STAGE42_FX_OBJECTIVE_COVERAGE_AUDIT:END -->

<!-- STAGE42_FY_HORIZON_RETRY_DECISION_MAP:START -->
## Stage42-FY Horizon Retry Decision Map

- source: `fresh_stage42_horizon_retry_decision_map_from_fl_fq`
- gate: `14 / 14`; verdict `stage42_fy_horizon_retry_decision_pass`.
- weak horizons: `['TrajNet|100', 'UCY|100']`.
- model retry attempts considered: `5`; promoted policy count `0`.
- decision: stop repeating same-feature weak-horizon model retries now = `True`.
- highest-priority unblocker: `FW-TERMS-ucy_crowd_original`.
- role: retry decision map for h100 weak slices; no training, no download, no conversion, no threshold tuning.
- boundary: uniform horizon robustness remains blocked; protected dataset-local/raw-frame 2.5D only; no metric/seconds, true 3D, foundation, Stage5C, or SMC claim.
<!-- STAGE42_FY_HORIZON_RETRY_DECISION_MAP:END -->

<!-- STAGE42_FZ_PAPER_PACKAGE_FXFY_REFRESH:START -->
## Stage42-FZ Paper Package FX/FY Refresh

- source: `fresh_stage42_paper_package_fxfy_refresh`
- gate: `20 / 20`; verdict `stage42_fz_paper_package_fxfy_refresh_pass`.
- role: paper-package refresh over Stage42-FX objective coverage and Stage42-FY horizon retry decision map; no training, no download, no conversion, no test-threshold tuning.
- supported core claims: `['history', 'domain_expert', 'safe_switch', 'teacher_floor', 'group_consistency_full_waypoint']`.
- blocked main claims: `['JEPA_downstream_lift', 'ungated_neural_dynamics', 'scene_goal_independent_main_claim', 'neighbor_interaction_independent_main_claim', 'global_metric_seconds_claim']`.
- objective status: blocked `['A']`, partial `['B', 'C', 'D']`, passed `['E']`, goal_complete `False`.
- weak horizons: `['TrajNet|100', 'UCY|100']`; stop_repeat_modeling_now `True`; uniform_horizon_claim_allowed `False`.
- highest-priority next action: `FW-TERMS-ucy_crowd_original`.
- boundary: protected dataset-local/raw-frame 2.5D only; no true 3D/foundation/global metric/seconds/Stage5C/SMC claim.
<!-- STAGE42_FZ_PAPER_PACKAGE_FXFY_REFRESH:END -->

<!-- STAGE42_GA_LIVE_SOURCE_CALIBRATION_RECHECK:START -->
## Stage42-GA Live Source / Calibration Recheck

- source: `fresh_stage42_live_source_calibration_recheck`
- gate: `15 / 15`; verdict `stage42_ga_live_source_calibration_recheck_pass`.
- role: fresh local path scan plus cached legal/calibration readiness recheck; no download, no conversion, no training, no evaluation.
- targets audited: `7`; local-path-found targets `7`; existing converted/cache targets `1`.
- new conversion-ready targets: `0`; source_action conversion_ready_now `0`; unified queue `0`.
- highest-priority next action: `FW-TERMS-ucy_crowd_original`.
- boundary: local file presence is not legal conversion readiness; protected dataset-local/raw-frame 2.5D only; no true 3D/foundation/global metric/seconds/Stage5C/SMC claim.
<!-- STAGE42_GA_LIVE_SOURCE_CALIBRATION_RECHECK:END -->

<!-- STAGE42_GB_SOURCE_TERMS_PREFILL:START -->
## Stage42-GB Source Terms Prefill

- source: `fresh_stage42_gb_source_terms_prefill`
- gate: `15 / 15`; verdict `stage42_gb_source_terms_prefill_pass`.
- role: converts Stage42-GA local path evidence into a user-facing source-terms prefill draft; no download, conversion, training, evaluation, or permission claim.
- datasets prefilled: `5`; with suggested local path `5`; raw-source candidates `5`.
- conversion_ready_now: `0`; highest-priority next action `FW-TERMS-ucy_crowd_original`.
- prefill draft: `outputs/stage42_long_research/source_terms_confirmation_prefill_stage42.json`.
- boundary: prefill is not legal permission; protected dataset-local/raw-frame 2.5D only; no true 3D/foundation/global metric/seconds/Stage5C/SMC claim.
<!-- STAGE42_GB_SOURCE_TERMS_PREFILL:END -->

<!-- STAGE42_GC_PREFILL_INTAKE_BRIDGE:START -->
## Stage42-GC Prefill -> Intake Bridge

- source: `fresh_stage42_gc_prefill_intake_bridge`
- gate: `16 / 16`; verdict `stage42_gc_prefill_intake_bridge_pass`.
- role: adds GB local path/source identity suggestions into the EH intake template as non-permission `prefill_suggestion` hints.
- intake rows: `5`; suggestions added `5`; user-confirmed rows `0`.
- conversion_ready_now: `0`; updated intake template `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`.
- boundary: user_confirmation is still blank; no download/conversion/training/evaluation; protected dataset-local/raw-frame 2.5D only.
<!-- STAGE42_GC_PREFILL_INTAKE_BRIDGE:END -->

<!-- STAGE42_GD_CALIBRATION_HINT_INTAKE_BRIDGE:START -->
## Stage42-GD Calibration Hint -> Intake Bridge

- source: `fresh_stage42_gd_calibration_hint_intake_bridge`
- gate: `18 / 18`; verdict `stage42_gd_calibration_hint_intake_bridge_pass`.
- role: adds DU metadata-only H/FPS/stride hints into the intake template as non-claim `calibration_prefill` leads.
- rows with hints: `3`; metric/time subset hint rows `2`.
- conversion_ready_now: `0`; metric/seconds claim allowed now `False` / `False`.
- boundary: hints are not permission, not conversion readiness, and not global metric/seconds evidence; Stage5C/SMC remain false.
<!-- STAGE42_GD_CALIBRATION_HINT_INTAKE_BRIDGE:END -->

<!-- STAGE42_GE_CONVERSION_CAPABILITY_INTAKE_BRIDGE:START -->
## Stage42-GE Conversion Capability -> Intake Bridge

- source: `fresh_stage42_ge_conversion_capability_intake_bridge`
- gate: `20 / 20`; verdict `stage42_ge_conversion_capability_intake_bridge_pass`.
- role: adds DW source-specific dry-run capability into the intake template as non-permission `conversion_capability_prefill`.
- source-specific rows available for `2` dataset rows; source-CV feasible after terms for `1` row.
- t50/t100 windows after terms: `10060` / `5696`; conversion_ready_now `0`.
- boundary: dry-run capability is not permission or conversion readiness; no download/conversion/training/evaluation; no metric/seconds/Stage5C/SMC claim.
<!-- STAGE42_GE_CONVERSION_CAPABILITY_INTAKE_BRIDGE:END -->

<!-- STAGE42_GF_POST_CONFIRMATION_CONVERSION_PLAN:START -->
## Stage42-GF Post-Confirmation Conversion Plan

- source: `fresh_stage42_gf_post_confirmation_conversion_plan`
- gate: `16 / 16`; verdict `stage42_gf_post_confirmation_conversion_plan_pass`.
- role: ranks GE source-specific conversion capability rows into a post-confirmation execution plan.
- planned source rows: `6`; technical-ready-after-terms sources `5`; source-CV-capable datasets `1`.
- t50/t100 after-terms windows: `10060` / `5696`; source_ready_now `0`; manifest ready targets `0`.
- EI validator recheck: `10 / 10`; FT unified guarded queue recheck: `12 / 12`, queue count `0`.
- verification: focused GF/GE/FT tests `11 passed`; full test suite `893 passed`.
- boundary: plan is not permission, not conversion, not evaluation; no metric/seconds/true-3D/foundation/Stage5C/SMC claim.
<!-- STAGE42_GF_POST_CONFIRMATION_CONVERSION_PLAN:END -->

<!-- STAGE42_GH_CALIBRATED_POST_CONFIRMATION_SUBSET_PLAN:START -->
## Stage42-GH Calibrated Post-Confirmation Subset Plan

- source: `fresh_stage42_gh_calibrated_post_confirmation_subset_plan`
- gate: `14 / 14`; verdict `stage42_gh_calibrated_post_confirmation_subset_plan_pass`.
- role: combines GF conversion planning with BN source-level H/FPS/geometry evidence.
- restricted metric/time candidates after terms: `5`; ready now `0`.
- calibrated after-terms t50/t100 windows: `10060` / `5696`.
- verification: focused GH/GF/BN/FT tests `15 passed`; full test suite `896 passed`.
- boundary: source-specific restricted subset candidate only after user terms + guarded conversion + no-leakage eval; global M3W remains raw-frame/dataset-local 2.5D.
<!-- STAGE42_GH_CALIBRATED_POST_CONFIRMATION_SUBSET_PLAN:END -->

<!-- STAGE42_GI_PAPER_CLAIM_EVIDENCE_REFRESH:START -->
## Stage42-GI Paper Claim Evidence Refresh With Calibrated Subset Plan

- source: `fresh_audit_from_stage42_wxy_paper_package_and_gh_calibrated_plan`
- gate: `25 / 25`; verdict `stage42_z_paper_claim_evidence_audit_pass`.
- role: refreshes the paper claim matrix with Stage42-GH calibrated post-confirmation subset candidates.
- C14 status: `post_confirmation_candidate_but_not_claimable`; main claim allowed: `False`.
- C14 evidence: restricted_candidates_after_terms=5; ready_now=0; calibrated_t50/t100=10060/5696; domains=['ETH_UCY', 'UCY']; converted/evaluated=0/0
- boundary: calibrated candidates are not converted/evaluated and cannot support current metric/seconds claims; no true-3D/foundation/Stage5C/SMC claim.
<!-- STAGE42_GI_PAPER_CLAIM_EVIDENCE_REFRESH:END -->

<!-- STAGE42_GJ_MODULE_CLAIM_LOCK:START -->
## Stage42-GJ Module Claim Lock

- source: `fresh_stage42_gj_module_claim_lock_from_fu_z_dp_dq_gh`
- gate: `19 / 19`; verdict `stage42_gj_module_claim_lock_pass`.
- locked supported modules: `['history', 'domain_expert', 'safe_switch', 'teacher_floor', 'group_consistency_full_waypoint', 'full_waypoint_shape', 'endpoint_bridge']`.
- locked blocked modules: `['scene_goal', 'neighbor_interaction', 'JEPA', 'Transformer']`.
- protected full-waypoint runtime supported: `True`; ungated full-waypoint deployable: `False`.
- calibrated post-confirmation candidates: `5`; ready now: `0`; after-terms t50/t100: `10060` / `5696`.
- next admissible experiments are restricted to terms-confirmed guarded conversion, changed-target gain/harm or full-sequence context, protected full-waypoint runtime replay, and source/horizon-specific h100 support repair.
- Still no true-3D, foundation, global metric, seconds-level, Stage5C, SMC, or post-confirmation-candidate-as-data claim.
<!-- STAGE42_GJ_MODULE_CLAIM_LOCK:END -->

<!-- STAGE42_GK_CONTEXT_SWITCHABILITY_FAMILY_AUDIT:START -->
## Stage42-GK Context Switchability Family Audit

- source: `fresh_stage42_gk_context_switchability_family_audit`
- gate: `14 / 14`; verdict `stage42_gk_context_switchability_family_audit_pass`.
- decision: `context_switchability_family_not_supported`; material context families: `[]`.
- best family `baseline_plus_history_goal_neighbor` vs baseline-family control: all/t50/t100raw/hard/easy = `-0.000003` / `0.000000` / `0.000000` / `0.000006` / `0.000093`.
- Target changed from residual trajectory deltas to gain/harm/switchability. Future labels are train/val/eval labels only, never inference inputs.
- If no material family is supported, scene/goal/neighbor context remains blocked as an independent main claim under this changed-target audit.
- Still no true-3D, foundation, global metric, seconds-level, Stage5C, SMC, or test-endpoint claim.
<!-- STAGE42_GK_CONTEXT_SWITCHABILITY_FAMILY_AUDIT:END -->

<!-- STAGE42_GL_SOURCE_CONVERSION_CONTRACT:START -->
## Stage42-GL Source Conversion Contract

- source: `fresh_stage42_gl_source_conversion_contract`
- role: locks the path from user terms/path/source identity confirmation to future guarded conversion.
- gate: `16 / 16`; verdict `stage42_gl_source_conversion_contract_pass`.
- intake datasets: `5`; manifest ready targets: `0`; contract ready now: `0`.
- post-confirmation calibrated source rows: `5`; calibrated t50/t100 opportunity after terms: `10060` / `5696`.
- No download, conversion, training, or evaluation was executed.
- Boundary: these candidates are not permission, converted data, metric/seconds claims, Stage5C, or SMC evidence.
<!-- STAGE42_GL_SOURCE_CONVERSION_CONTRACT:END -->

<!-- STAGE42_GM_GUARDED_CONVERSION_HARNESS:START -->
## Stage42-GM Guarded Conversion Harness

- source: `fresh_stage42_gm_guarded_conversion_harness`
- role: executable barrier for future source-specific conversion; current run is dry-run and refuses conversion because no contract row is ready.
- gate: `14 / 14`; verdict `stage42_gm_guarded_conversion_harness_pass`.
- contract_ready_now: `0`; execution_plan_count: `0`; blocked_contract_rows: `5`.
- No download, conversion, feature-store build, no-leakage audit, source-CV, training, or evaluation was executed.
- Boundary: this is not converted data, not metric/seconds evidence, not Stage5C, and not SMC.
<!-- STAGE42_GM_GUARDED_CONVERSION_HARNESS:END -->

<!-- STAGE42_GN_SOURCE_CONFIRMATION_PRIORITY_BOARD:START -->
## Stage42-GN Source Confirmation Priority Board

- source: `fresh_stage42_gn_source_confirmation_priority_board`
- role: ranks user-confirmation actions needed before any guarded source conversion can legally run.
- gate: `14 / 14`; verdict `stage42_gn_source_confirmation_priority_board_pass`.
- targets_ranked: `5`; ready_now: `0`; blocked_now: `5`.
- top priority: `ucy_crowd_original` / `UCY`; value class `calibrated_t50_t100_unlock`.
- after-terms opportunity: t50 `10060`, t100 `5696`; calibrated t50/t100 `10060` / `5696`.
- No download, conversion, feature-store build, no-leakage audit, source-CV, training, or evaluation was executed.
- Boundary: this is a source/legal/calibration unblock queue only; not converted data, not metric/seconds evidence, not Stage5C, and not SMC.
<!-- STAGE42_GN_SOURCE_CONFIRMATION_PRIORITY_BOARD:END -->

<!-- STAGE42_GO_OFFICIAL_SOURCE_TERMS_LIVE_VERIFIER:START -->
## Stage42-GO Official Source / Terms Live Verifier

- source: `fresh_stage42_go_official_source_terms_live_verifier`
- role: official source/terms live audit for the Stage42-GN priority queue; it does not accept terms or download data.
- gate: `14 / 14`; verdict `stage42_go_official_source_terms_live_verifier_pass`.
- datasets_audited: `5`; official_sources_reachable: `3`; auto_download_allowed_now: `0`.
- top priority remains `ucy_crowd_original`; terms status `not_verified_by_agent`.
- OpenTraj toolkit license is explicitly not counted as underlying dataset permission.
- No download, conversion, feature-store build, training, or evaluation was executed.
- Boundary: source/terms audit only; not converted data, not metric/seconds evidence, not Stage5C, and not SMC.
<!-- STAGE42_GO_OFFICIAL_SOURCE_TERMS_LIVE_VERIFIER:END -->

<!-- STAGE42_GP_SOURCE_TERMS_PAPER_CLAIM_GUARD:START -->
## Stage42-GP Source Terms Paper Claim Guard

- source: `fresh_stage42_gp_source_terms_paper_claim_guard`
- role: writes the GO source/terms blocker into data card, method draft, and A-journal gap so paper claims cannot overrun legal/source evidence.
- gate: `12 / 12`; verdict `stage42_gp_source_terms_paper_claim_guard_pass`.
- paper files refreshed: `3`; unsafe source-claim violations: `0`.
- No source is license-confirmed, auto-downloadable, conversion-ready, converted, trained, or evaluated by this step.
- Boundary: source/terms paper guard only; not converted data, not metric/seconds evidence, not Stage5C, and not SMC.
<!-- STAGE42_GP_SOURCE_TERMS_PAPER_CLAIM_GUARD:END -->

<!-- STAGE42_GQ_SOURCE_TERMS_PACKAGE_CLAIM_LINTER:START -->
## Stage42-GQ Source Terms Package Claim Linter

- source: `fresh_stage42_gq_source_terms_package_claim_linter`
- role: scans README and Stage42 paper package for source/legal overclaims after GO/GP.
- gate: `13 / 13`; verdict `stage42_gq_source_terms_package_claim_linter_pass`.
- files scanned: `14`; violations: `0`.
- No source is license-confirmed, auto-downloadable, conversion-ready, converted, trained, or evaluated by this step.
- Boundary: package-wide claim lint only; not converted data, not metric/seconds evidence, not Stage5C, and not SMC.
<!-- STAGE42_GQ_SOURCE_TERMS_PACKAGE_CLAIM_LINTER:END -->

<!-- STAGE42_GR_REFRESH:START -->
## Stage42-GR Long Objective State Reconciler

- source: `fresh_stage42_gr_long_objective_state_reconciler`
- verdict: `stage42_gr_long_objective_state_reconciler_pass`
- gates: `14 / 14`
- objectives reconciled: `6`
- contract ready now: `0`
- auto-download allowed now: `0`
- package source-claim violations: `0`
- after-terms opportunity: t50 `10060`, t100 `5696`
- This is a fresh reconciliation step, not a data/model execution step.
- Current deployable status remains protected dataset-local/raw-frame 2.5D candidate; no true 3D, no foundation, no global metric/seconds-level, no Stage5C, no SMC.
<!-- STAGE42_GR_REFRESH:END -->

<!-- STAGE42_GS_REFRESH:START -->
## Stage42-GS Paper Gap Reconciler

- source: `fresh_stage42_gs_paper_gap_reconciler`
- verdict: `stage42_gs_paper_gap_reconciler_pass`
- gates: `13 / 13`
- gap rows: `5`
- stale findings reconciled: `4`
- open blockers: `['source_legal_conversion', 'floor_free_neural_deployment', 'paper_package_source_claim_safety']`
- It refreshes A-journal gap language against current module, floor, source/legal, and package-claim guards.
- No download, conversion, training, or evaluation was executed.
<!-- STAGE42_GS_REFRESH:END -->

<!-- STAGE42_GU_FLOOR_RELAXATION_SAFETY_REFRESH:START -->
## Stage42-GU Floor Relaxation Safety Refresh

- source: `fresh_stage42_gu_floor_relaxation_paper_refresh`
- role: propagates Stage42-GT all-agent safety stress evidence into the paper package and guards against floor overclaims.
- input GT verdict: `stage42_gt_floor_relaxation_safety_stress_pass`; input BY/BZ/EN gates passed: `True` / `True` / `True`.
- target union t50 rows: `11538`.
- target union t50 improvement: `28.97%`.
- target union hard/failure improvement: `28.97%`.
- target union easy degradation: `-21.41%`.
- target union near-collision@0.05 delta: `-0.74%`.
- target union jagged-rate delta: `0.00%`.
- Supported claim: narrow validation-backed t50 partial floor relaxation has all-agent safety support for the audited slices.
- Unsupported claims: global floor removal, floor-free neural deployment, teacher/floor context removal, metric/seconds-level prediction, Stage5C execution, and SMC readiness.
- Result source label: `fresh_run` synthesis from already-produced Stage42-BY/BZ/EN/GT artifacts; no new training, no new download, no new conversion, no test threshold tuning.
- Verification after implementation: focused pytest passed; full suite passed with `929 passed`.
<!-- STAGE42_GU_FLOOR_RELAXATION_SAFETY_REFRESH:END -->

<!-- STAGE42_GV_FLOOR_RELAXATION_SOURCE_ROBUSTNESS:START -->
## Stage42-GV Floor Relaxation Source Robustness

- source: `fresh_stage42_gv_floor_relaxation_source_robustness`
- role: source-level all-agent robustness audit for Stage42-GT partial t50 floor relaxation.
- gate: `14 / 14`; verdict `stage42_gv_floor_relaxation_source_robustness_pass_with_source_concentration_caveat`.
- source-safety-positive slices: `['TrajNet|50', 'UCY|50']`.
- source-concentration-limited slices: `['TrajNet|50', 'UCY|50']`.
- broad source-level generalization claim allowed: `False`.
- Claim boundary: major-source support only; not broad source-level generalization, not global floor removal, not floor-free neural, not metric/seconds-level, not Stage5C, not SMC.
<!-- STAGE42_GV_FLOOR_RELAXATION_SOURCE_ROBUSTNESS:END -->

<!-- STAGE42_GW_H100_BLOCKER_CLOSURE_DECISION:START -->
## Stage42-GW H100 Blocker Closure Decision

- source: `fresh_stage42_gw_h100_blocker_closure_decision`
- gate: `17 / 17`; verdict `stage42_gw_h100_blocker_closure_decision_pass`
- weak keys: `['TrajNet|100', 'UCY|100']`
- technical support exists count: `1`; legal conversion ready count: `0`; can run repair now count: `0`
- `UCY|100`: local technical candidates exist, but terms/source identity/guarded conversion are not ready; user action required before repair.
- `TrajNet|100`: hard blocker remains because current local TrajNet snippets are too short for raw-frame h100/t100 repair.
- Boundary: no download, no conversion, no training, no evaluation; uniform h100/t100 claim remains blocked; no metric/seconds, no Stage5C, no SMC.
<!-- STAGE42_GW_H100_BLOCKER_CLOSURE_DECISION:END -->

<!-- STAGE42_GX_UCY_H100_CANDIDATE_INTEGRITY:START -->
## Stage42-GX UCY H100 Candidate Integrity Manifest

- source: `fresh_stage42_gx_ucy_h100_candidate_integrity_manifest`
- gate: `17 / 17`; verdict `stage42_gx_ucy_h100_candidate_integrity_manifest_pass`
- UCY candidate files: `6`; existing `6`; target-family candidates `2`.
- parsed rows: `98032`; parsed t100 windows: `11848`; unique hashes `6`.
- This locks file identity/hash/parse stats only. It is not legal permission, not conversion, not evaluation, and not h100 repair.
- `UCY|100` remains terms/source-identity blocked; `TrajNet|100` remains long-source blocked. No metric/seconds, no Stage5C, no SMC.
<!-- STAGE42_GX_UCY_H100_CANDIDATE_INTEGRITY:END -->

<!-- STAGE42_GY_UCY_H100_TERMS_PREFILL:START -->
## Stage42-GY UCY H100 Terms Prefill From Integrity

- source: `fresh_stage42_gy_ucy_h100_terms_prefill_from_integrity`
- gate: `14 / 14`; verdict `stage42_gy_ucy_h100_terms_prefill_pass`
- prefill rows: `6`; rows with hash/source identity suggestions: `6` / `6`.
- Legal acceptance fields remain blank and must be user-confirmed. This is not conversion, evaluation, h100 repair, metric evidence, or seconds-level evidence.
- `UCY|100` remains blocked until terms/path/source identity are confirmed and guarded conversion/no-leakage/source-CV pass.
<!-- STAGE42_GY_UCY_H100_TERMS_PREFILL:END -->

<!-- STAGE42_GZ_FULL_WAYPOINT_CLAIM_GUARD:START -->
## Stage42-GZ Full-Waypoint Claim Guard

- source: `fresh_stage42_gz_full_waypoint_claim_guard`
- gate: `18 / 18`
- verdict: `stage42_gz_full_waypoint_claim_guard_pass`
- Protected full-waypoint evidence can be cited only as dataset-local/raw-frame 2.5D evidence.
- Endpoint-only or endpoint-linear bridge success must not be counted as learned full-waypoint dynamics.
- Ungated full-waypoint neural deployment remains rejected.
- Group-consistency full-waypoint is supported under protected policy; neighbor/interaction alone remains blocked as an independent main claim.
- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.
<!-- STAGE42_GZ_FULL_WAYPOINT_CLAIM_GUARD:END -->

<!-- STAGE42_HA_FULL_WAYPOINT_OVERCLAIM_LINTER:START -->
## Stage42-HA Full-Waypoint Overclaim Linter

- source: `fresh_stage42_ha_full_waypoint_overclaim_linter`
- gate: `14 / 14`
- verdict: `stage42_ha_full_waypoint_overclaim_linter_pass`
- files_scanned: `15`
- violations_total: `0`
- Endpoint/full-waypoint, ungated full-waypoint, group/neighbor independent-main, metric/seconds, Stage5C and SMC overclaims were scanned.
- No unsupported full-waypoint overclaim lines were found.
<!-- STAGE42_HA_FULL_WAYPOINT_OVERCLAIM_LINTER:END -->

<!-- STAGE42_HB_TEACHER_FLOOR_NECESSITY_META_AUDIT:START -->
## Stage42-HB Teacher-Floor Necessity Meta-Audit

- source: `fresh_stage42_hb_teacher_floor_necessity_meta_audit`
- gate: `16 / 16`
- verdict: `stage42_hb_teacher_floor_necessity_meta_audit_pass`
- Direct conclusion: Stage37 / teacher floor is the current safety mechanism and rollout-context floor, not merely a disposable crutch.
- Protected current all/t50/t100raw/hard/easy: `21.03%` / `13.65%` / `14.69%` / `20.38%` / `0.00%`.
- Ungated endpoint/full-waypoint easy degradation remains unsafe: `124.59%` / `124.59%`.
- Narrow t50 floor relaxation is supported only on selected slices: rows `11538`, t50 `28.97%`, hard `28.97%`, easy `-21.41%`.
- Global floor removal and floor-free neural deployment remain false.
- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.
<!-- STAGE42_HB_TEACHER_FLOOR_NECESSITY_META_AUDIT:END -->

<!-- STAGE42_HC_FLOOR_ALTERNATIVE_GATE_STRESS:START -->
## Stage42-HC Floor-Alternative Gate Stress Matrix

- source: `fresh_stage42_hc_floor_alternative_gate_stress`
- gate: `14 / 14`
- verdict: `stage42_hc_floor_alternative_gate_stress_pass`
- Tested Stage42-E internal self-gate, uncertainty gate, conformal risk gate, harm predictor, teacher-dependent gates, and bounded residual families as floor alternatives.
- floor-free deployable count: `0`; teacher-dependent deployable count: `6`.
- best floor-free candidate `harm_predictor_gate` reaches all/t50/hard `35.95%` / `25.20%` / `35.86%` but is not deployable because `['near_collision_delta_over_1pp']`.
- best deployable teacher-dependent candidate `current_composite_tail_policy` reaches all/t50/hard `21.03%` / `13.65%` / `20.38%` with easy `0.00%`.
- Deployment decision remains: keep Stage37/teacher floor globally; allow only validation-backed partial t50 relaxation on selected slices.
- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.
<!-- STAGE42_HC_FLOOR_ALTERNATIVE_GATE_STRESS:END -->

<!-- STAGE42_HD_FLOOR_FREE_PROXIMITY_GUARD_REPAIR:START -->
## Stage42-HD Floor-Free Proximity-Guard Repair

- source: `fresh_stage42_hd_floor_free_proximity_guard_repair`
- gate: `13 / 13`
- verdict: `stage42_hd_floor_free_proximity_guard_repair_pass`
- Tested floor-free internal/harm/uncertainty/conformal gates with a validation-selected proximity guard.
- pre-guard deployable count: `0`; post-guard deployable count: `4`.
- best post-guard family `harm_predictor_gate` reaches all/t50/t100raw/hard `20.74%` / `13.82%` / `13.68%` / `19.99%` with easy `0.00%` and collision delta `-0.47%`.
- The teacher gate is not used in this repair, but causal floor fallback remains required; this is not global floor removal.
- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.
<!-- STAGE42_HD_FLOOR_FREE_PROXIMITY_GUARD_REPAIR:END -->

<!-- STAGE42_HE_FLOOR_FREE_PROXIMITY_GUARD_ROBUSTNESS:START -->
## Stage42-HE Floor-Free Proximity-Guard Robustness Audit

- source: `fresh_stage42_he_floor_free_proximity_guard_robustness`
- gate: `21 / 21`
- verdict: `stage42_he_floor_free_proximity_guard_robustness_pass`
- Audits the Stage42-HD teacherless proximity-guard repaired gate with 2000-bootstrap and per-domain/per-horizon checks.
- policy `harm_predictor_gate` with min_sep `0.05` reaches all/t50/t100raw/hard `20.74%` / `13.82%` / `13.68%` / `19.99%`.
- bootstrap CI lows all/t50/t100raw/hard `20.38%` / `13.22%` / `12.94%` / `19.57%`; easy CI high `-16.17%`.
- robust_positive_domains: `ETH_UCY, TrajNet, UCY`; weak_domain_horizon_slices: `none`.
- Teacher gate is not used, but causal floor fallback remains required. This is not global floor removal, not metric/seconds, not true 3D, not Stage5C, and not SMC.
<!-- STAGE42_HE_FLOOR_FREE_PROXIMITY_GUARD_ROBUSTNESS:END -->

<!-- STAGE42_HF_TEACHERLESS_GATE_DEPLOYMENT_CONTRACT:START -->
## Stage42-HF Teacherless Gate Deployment Contract

- source: `fresh_stage42_hf_teacherless_gate_deployment_contract`
- verdict: `stage42_hf_teacherless_gate_deployment_contract_pass`
- gates: `15 / 15`
- result: Stage42-HE supports a teacherless proximity-guarded switch gate, but only with causal floor fallback.
- metrics: all `20.74%`, t50 `13.82%`, t100 raw diagnostic `13.68%`, hard/failure `19.99%`, easy degradation `0.00%`.
- allowed claim: `teacherless proximity-guarded switch gate with causal floor fallback`.
- blocked claims: global causal floor removal, ungated neural deployment, metric/seconds/true-3D/foundation claims, Stage5C execution, and SMC.
- deployment default remains protected causal-floor fallback; Stage42-HF is a claim/deployment contract refresh, not new training.
<!-- STAGE42_HF_TEACHERLESS_GATE_DEPLOYMENT_CONTRACT:END -->

<!-- STAGE42_HG_TEACHERLESS_CLAIM_LINTER:START -->
## Stage42-HG Teacherless / Floor-Free Claim Linter

- source: `fresh_stage42_hg_teacherless_claim_linter`
- verdict: `stage42_hg_teacherless_claim_linter_pass`
- gates: `15 / 15`
- scanned files: `18`; violations: `0`.
- allowed phrase: `teacherless proximity-guarded switch gate with causal floor fallback`.
- blocked: global floor-free neural deployment, causal floor removal, ungated neural deployment, metric/seconds/true-3D/foundation claims, Stage5C, and SMC.
- role: applies Stage42-HF contract to the paper/README surface; this is not new training or threshold tuning.
<!-- STAGE42_HG_TEACHERLESS_CLAIM_LINTER:END -->

<!-- STAGE42_HI_RESTRICTED_METRIC_TIME_READINESS:START -->
## Stage42-HI Restricted Metric/Time Readiness

- source: `fresh_stage42_hi_restricted_metric_time_readiness`
- verdict: `stage42_hi_restricted_metric_time_readiness_pass_blocked_by_terms`
- gates: `14 / 14`
- restricted metric/time candidates: `6` across `['ETH_UCY', 'UCY']`.
- technical ready after terms: `6`; ready now: `0`.
- conclusion: ETH/UCY source-level H/FPS/stride evidence exists, but no metric/seconds claim is allowed until user-confirmed source terms plus conversion/no-leakage/source-CV/final-test.
- no training, conversion, download, Stage5C, or SMC occurred.
<!-- STAGE42_HI_RESTRICTED_METRIC_TIME_READINESS:END -->

<!-- STAGE42_HJ_RESTRICTED_METRIC_TIME_SOURCE_CV_PREFLIGHT:START -->
## Stage42-HJ Restricted Metric/Time Source-CV Preflight

- source: `fresh_stage42_hj_restricted_metric_time_source_cv_preflight`
- verdict: `stage42_hj_restricted_metric_time_source_cv_preflight_pass_with_eth_ucy_source_cv_limit`
- gates: `15 / 15`
- usable after terms sources: `4`; ready now: `0`.
- source-CV feasible after terms: `['UCY']`; robust after terms: `['UCY']`.
- source-CV blocked after terms: `['ETH_UCY']`.
- window potential after terms: t50 `9845`, t100 `5696`.
- conclusion: restricted metric/time source-CV is technically plannable for UCY and blocked for ETH_UCY by current t100 source support; source terms still block all conversion/evaluation claims.
<!-- STAGE42_HJ_RESTRICTED_METRIC_TIME_SOURCE_CV_PREFLIGHT:END -->

<!-- STAGE42_HK_ETH_UCY_SOURCE_SUPPORT_PREFLIGHT:START -->
## Stage42-HK ETH_UCY Restricted Metric/Time Source-Support Preflight

- source: `fresh_stage42_hk_restricted_metric_time_eth_ucy_source_support_preflight`
- verdict: `stage42_hk_eth_ucy_source_support_preflight_pass_terms_blocked`
- gates: `16 / 16`
- augmented ETH_UCY independent sources after terms: `5`.
- augmented ETH_UCY t50/t100 windows after terms: `4397` / `1433`.
- cached BL technical t100 safe-positive: `True`; ready now: `False`.
- conclusion: ETH_UCY source-CV blocker is technically repairable after terms using ETH-Person XML candidates, but conversion/evaluation and metric/seconds claims remain blocked until user-confirmed terms and guarded rerun.
<!-- STAGE42_HK_ETH_UCY_SOURCE_SUPPORT_PREFLIGHT:END -->

<!-- STAGE42_HL_RESTRICTED_METRIC_TIME_POST_HK_CLAIM_GUARD:START -->
## Stage42-HL Restricted Metric/Time Post-HK Claim Guard

- source: `fresh_stage42_hl_restricted_metric_time_post_hk_claim_guard`
- verdict: `stage42_hl_restricted_metric_time_post_hk_claim_guard_pass`
- gates: `15 / 15`
- files scanned / violations: `14` / `0`.
- HK after-terms source support: `5` sources, t50/t100 windows `4397` / `1433`.
- ready now: `False`; conversion ready targets now: `0`.
- conclusion: the paper/README package remains claim-safe after HK; ETH_UCY source support is technically repairable after terms, but restricted metric/time conversion/evaluation remains blocked until user confirmation and guarded rerun.
<!-- STAGE42_HL_RESTRICTED_METRIC_TIME_POST_HK_CLAIM_GUARD:END -->

<!-- STAGE42_HM_RESTRICTED_METRIC_TIME_TERMS_INTAKE_V2:START -->
## Stage42-HM Restricted Metric/Time Terms Intake v2

- source: `fresh_stage42_hm_restricted_metric_time_terms_intake_v2`
- verdict: `stage42_hm_restricted_metric_time_terms_intake_v2_pass_blocked_until_user_confirmation`
- gates: `15 / 15`
- source-level candidates / ready now: `11` / `0`.
- after-terms domains: `{'UCY': 3, 'ETH_UCY': 6}`.
- after-terms t50/t100 windows: `14457` / `7129`.
- template: `outputs/stage42_long_research/restricted_metric_time_terms_intake_v2_template_stage42.json`.
- conclusion: UCY/ETH_UCY restricted metric/time source-level candidates are now represented in a user-fillable intake v2, but all conversion/evaluation remains blocked until user-confirmed terms/source identity/path and a guarded rerun.
<!-- STAGE42_HM_RESTRICTED_METRIC_TIME_TERMS_INTAKE_V2:END -->

<!-- STAGE42_HN_RESTRICTED_METRIC_TIME_CONVERSION_QUEUE_V2:START -->
## Stage42-HN Restricted Metric/Time Conversion Queue v2

- source: `fresh_stage42_hn_restricted_metric_time_conversion_queue_v2`
- verdict: `stage42_hn_restricted_metric_time_conversion_queue_v2_pass_blocked_until_ready_candidates`
- gates: `15 / 15`
- ready / blocked candidates: `0` / `11`.
- conversion queue count: `0`.
- blocked after-terms t50/t100 windows retained: `14457` / `7129`.
- conclusion: the restricted metric/time execution path is now guarded by HM ready-candidate validation; current conversion remains refused until user-confirmed terms/source identity/path are supplied.
<!-- STAGE42_HN_RESTRICTED_METRIC_TIME_CONVERSION_QUEUE_V2:END -->

<!-- STAGE42_HO_LONG_OBJECTIVE_AUDIT:START -->
## Stage42-HO Long Research Objective Audit

本轮继续 Stage42 Long Research Mode，新增长期目标覆盖审计：

`/Users/yangyue/Downloads/World/outputs/stage42_long_research/long_research_objective_audit_stage42.md`

结果来源：`fresh_stage42_ho_long_research_objective_audit`；gate `17 / 17`；verdict `stage42_ho_long_research_objective_audit_pass_keep_goal_active`。
该审计不下载、不转换、不训练、不调 test threshold；它把 Stage42 A-F 要求映射到当前 authoritative evidence，并明确保持长期目标 active。

结论：external/protected full-waypoint/group-consistency evidence 已经很强，但 full objective 尚未完成。metric/time conversion 仍因 ready candidates = 0 被阻塞；JEPA、scene/goal、neighbor/interaction 独立主 claim 仍不支持；Stage5C 与 SMC 仍禁止。

<!-- STAGE42_HO_LONG_OBJECTIVE_AUDIT:END -->

<!-- STAGE42_HP_GROUP_CONSISTENCY_BREAKDOWN:START -->
## Stage42-HP Group-Consistency Source Breakdown

- source: `fresh_run_group_consistency_source_breakdown`
- role: break down frozen group-consistency full-waypoint policy by domain/source/scene/horizon/subset.
- gate: `23 / 23`; verdict `stage42_hp_group_consistency_breakdown_pass`.
- rows: `47458`; domains `{'TrajNet': 37918, 'UCY': 9540}`.
- ADE vs train-horizon causal floor: all `24.72%`, t50 `22.36%`, t100 raw `14.35%`, hard `23.89%`, easy `-25.63%`.
- FDE: all `22.29%`, t50 `22.57%`, t100 raw `12.85%`.
- group safety near@0.05 delta vs base: `-0.55%`.
- weak slices recorded: `6`; top examples `['domain:UCY', 'source:UCY::TrajNet/Train/crowds/crowds_zara03.txt', 'scene:UCY::UCY_crowds', 'fallback_only', 'horizon:100']`.
- claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_HP_GROUP_CONSISTENCY_BREAKDOWN:END -->

<!-- STAGE42_HQ_GROUP_CONSISTENCY_WEAK_SLICE_REPAIR:START -->
## Stage42-HQ UCY Weak-Slice Group-Consistency Repair

- source: `fresh_ucy_internal_validation_supported_repair`
- role: repair the Stage42-HP UCY zero-gain weak slice with train-only UCY internal validation support.
- gate: `23 / 23`; verdict `stage42_hq_group_consistency_weak_slice_repair_pass`.
- HP UCY before: all `0.00%`, t50 `0.00%`.
- repaired global all/t50/t100 raw/hard/easy: `32.89%` / `26.99%` / `21.12%` / `31.89%` / `-32.09%`.
- repaired UCY all/t50/hard/easy: `35.58%` / `22.72%` / `33.78%` / `-40.60%`.
- t100 easy status: rows `975`, degradation `2.56%`; recorded as raw-frame diagnostic, not seconds-level.
- claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_HQ_GROUP_CONSISTENCY_WEAK_SLICE_REPAIR:END -->

<!-- STAGE42_HR_GROUP_CONSISTENCY_T100_EASY_GUARD:START -->
## Stage42-HR Group-Consistency T100 Easy Guard

- source: `fresh_validation_only_domain_t100_easy_guard`
- role: repair Stage42-HQ t100 easy degradation with validation-only domain|t100 fallback decisions.
- gate: `23 / 23`; verdict `stage42_hr_t100_easy_guard_pass`.
- HQ t100 easy before: `2.56%`; after guard `-0.31%`.
- guarded all/t50/t100 raw/hard/easy: `27.72%` / `26.99%` / `6.79%` / `25.93%` / `-32.33%`.
- guarded slices: `{'TrajNet|100': {'source': 'fresh_validation_only_domain_t100_easy_guard', 'domain': 'TrajNet', 'val_rows': 1160, 'test_rows': 5608, 'val_all_improvement': 0.23260462520508085, 'val_easy_degradation': 0.017118176622190173, 'threshold': 0.0, 'keep': False, 'reason': 'validation_easy_degradation_above_threshold_or_nonpositive_gain'}}`; kept slices: `{'UCY|100': {'source': 'fresh_validation_only_domain_t100_easy_guard', 'domain': 'UCY', 'val_rows': 1440, 'test_rows': 1440, 'val_all_improvement': 0.27564518723015075, 'val_easy_degradation': -0.021788147627511134, 'threshold': 0.0, 'keep': True}}`.
- claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_HR_GROUP_CONSISTENCY_T100_EASY_GUARD:END -->

<!-- STAGE42_HS_T100_EASY_GUARD_FREEZE:START -->
## Stage42-HS Frozen T100 Easy Guard

- source: `cached_verified_stage42_hr_policy_freeze_from_fresh_artifact`
- role: freeze Stage42-HR validation-only domain|t100 easy guard as a lightweight policy/replay artifact.
- policy artifact: `outputs/stage42_long_research/frozen_group_consistency_t100_easy_guard_policy_stage42.json`
- policy hash: `8dcc60f145df211084868a57b57246b69364adf51add1578c88cd012a6121e6e`
- gate: `27 / 27`; verdict `stage42_hs_t100_easy_guard_freeze_pass`.
- replay: decision table exact `True`, metric summary exact `True`.
- guarded all/t50/t100 raw/hard/easy: `27.72%` / `26.99%` / `6.79%` / `25.93%` / `-32.33%`.
- t100 easy degradation after guard: `-0.31%`.
- Claim boundary: protected dataset-local/raw-frame 2.5D only; t100 remains raw-frame diagnostic; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_HS_T100_EASY_GUARD_FREEZE:END -->

<!-- STAGE42_HT_T100_EASY_GUARD_RUNTIME:START -->
## Stage42-HT Runtime T100 Easy Guard Policy

- source: `fresh_runtime_api_from_frozen_stage42_hs_t100_easy_guard_policy`
- role: convert the frozen Stage42-HS domain|t100 easy guard into a callable runtime policy API.
- gate: `19 / 19`; verdict `stage42_ht_t100_easy_guard_runtime_policy_pass`.
- policy artifact: `outputs/stage42_long_research/frozen_group_consistency_t100_easy_guard_policy_stage42.json`
- policy hash: `8dcc60f145df211084868a57b57246b69364adf51add1578c88cd012a6121e6e`
- runtime rule: TrajNet|100 falls back to floor; UCY|100 keeps candidate; unknown t100 domains fallback to floor; non-t100 rows are unchanged.
- inherited guarded all/t50/t100 raw/hard/easy: `27.72%` / `26.99%` / `6.79%` / `25.93%` / `-32.33%`.
- Claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_HT_T100_EASY_GUARD_RUNTIME:END -->

<!-- STAGE42_HU_T100_RUNTIME_BATCH_REPLAY_SUFFICIENCY:START -->
## Stage42-HU T100 Runtime Batch Replay Sufficiency Audit

- source: `fresh_audit_from_stage42_hr_hs_ht_artifacts`
- role: audit whether the frozen/runtime t100 easy guard evidence supports real row-level batch replay.
- gate: `17 / 17`; verdict `stage42_hu_t100_runtime_batch_replay_sufficiency_pass_with_blocker`.
- runtime API ready: `True`; frozen policy ready: `True`.
- real batch replay status: `not_run`; blocker: `missing_row_level_candidate_floor_selected_arrays`.
- conclusion: HT is callable and smoke-tested, but not a real batch replay because row-level candidate/floor/selected arrays are absent from HR/HS/HT artifacts.
- inherited guarded all/t50/t100 raw/hard/easy: `27.72%` / `26.99%` / `6.79%` / `25.93%` / `-32.33%`.
- Claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_HU_T100_RUNTIME_BATCH_REPLAY_SUFFICIENCY:END -->

<!-- STAGE42_HV_T100_RUNTIME_ROW_CACHE_REPLAY:START -->
## Stage42-HV T100 Runtime Row-Cache Batch Replay

- source: `fresh_or_cached_row_cache_reconstruction_and_runtime_batch_replay_from_stage42_hr_ht`
- role: close the Stage42-HU blocker by reconstructing a local row-level cache and replaying the frozen Stage42-HT runtime policy over full test rows.
- gate: `28 / 28`; verdict `stage42_hv_t100_runtime_row_cache_replay_pass`.
- cache path: `data/stage42_t100_runtime_replay_cache/stage42hv_t100_runtime_replay_test_cache.npz` (derived local data; not committed).
- cache hash: `166fdede23d8f14bbf6eb4c0398b32b9c90d489a03d3e6e9acdbc608db5ed127`.
- runtime replay rows/domains/t100 rows: `47458` / `{'TrajNet': 37918, 'UCY': 9540}` / `7048`.
- replay all/t50/t100 raw/hard/easy: `27.72%` / `26.99%` / `6.79%` / `25.93%` / `-32.33%`.
- t100 easy degradation: `-0.31%`.
- Claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_HV_T100_RUNTIME_ROW_CACHE_REPLAY:END -->

<!-- STAGE42_HW_REPLAY_EVIDENCE_TIER_REFRESH:START -->
## Stage42-HW Replay Evidence Tier Refresh

- source: `fresh_replay_evidence_tier_refresh_from_stage42_hs_ht_hu_hv`
- role: integrate HS/HT/HU/HV replay levels into reviewer replay and paper evidence matrix.
- gate: `30 / 30`; verdict `stage42_hw_replay_evidence_tier_refresh_pass`.
- evidence tiers: T1 runtime smoke, T2 frozen metric replay, T2.5 blocker audit, T3 row-level batch replay.
- T3 row-level rows/t100 rows: `47458` / `7048`.
- T3 all/t50/t100 raw/hard/easy: `27.72%` / `26.99%` / `6.79%` / `25.93%` / `-32.33%`.
- Claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_HW_REPLAY_EVIDENCE_TIER_REFRESH:END -->

<!-- STAGE42_HX_PAPER_PACKAGE_INTEGRITY:START -->
## Stage42-HX Paper Package Integrity

- source: `fresh_stage42_hx_paper_package_integrity_from_current_artifacts`
- role: verify Stage42 paper package deliverables, evidence provenance, replay-tier linkage, and claim boundaries.
- gate: `25 / 25`; verdict `stage42_hx_paper_package_integrity_pass`.
- paper deliverables checked: `9`.
- support/evidence files checked: `10`.
- A-F objective coverage is preserved as: A partial/blocked, B/C/E pass-with-boundary, D mixed, F pass-with-open-gaps.
- Claim boundary: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.
<!-- STAGE42_HX_PAPER_PACKAGE_INTEGRITY:END -->

<!-- STAGE42_HY_SOURCE_LOCAL_PATH_PREFILL:START -->
## Stage42-HY Source Local Path Prefill

- source: `fresh_stage42_hy_source_local_path_prefill_from_local_files`
- role: reduce source/legal blocker by pre-filling local path and parseability candidates without claiming legal conversion.
- gate: `19 / 19`; verdict `stage42_hy_source_local_path_prefill_pass`.
- targets audited: `5`; local path candidates found: `5`.
- estimated after-terms t50/t100 windows preserved: `10060` / `5696`.
- Remaining blocker: user must confirm official terms, allowed use, acceptance date, local path, and source identity before guarded conversion.
- Claim boundary: no download, no conversion, no training, no evaluation, no metric/seconds-level claim, no Stage5C, no SMC.
<!-- STAGE42_HY_SOURCE_LOCAL_PATH_PREFILL:END -->

<!-- STAGE42_HZ_SOURCE_TERMS_CONFIRMATION_PACKET:START -->
## Stage42-HZ Source Terms Confirmation Packet

- source: `fresh_stage42_hz_source_terms_confirmation_packet_from_hy_prefill`
- role: turn HY local path prefill into a user-confirmable source/terms packet and readiness validator.
- gate: `22 / 22`; verdict `stage42_hz_source_terms_confirmation_packet_pass`.
- sources in packet: `5`; ready for guarded conversion now: `0`.
- potential after-terms t50/t100 windows preserved: `10060` / `5696`.
- Remaining blocker: user must fill and confirm terms/source/local-path fields before conversion.
- Claim boundary: no download, no conversion, no training, no evaluation, no metric/seconds-level claim, no Stage5C, no SMC.
<!-- STAGE42_HZ_SOURCE_TERMS_CONFIRMATION_PACKET:END -->

<!-- STAGE42_IA_HZ_TO_CG_INTAKE_BRIDGE:START -->
## Stage42-IA HZ to CG Intake Bridge

- source: `fresh_stage42_ia_hz_to_cg_intake_bridge`
- role: bridge the new HZ confirmation packet into the older CG validator intake schema without activating conversion.
- gate: `17 / 17`; verdict `stage42_ia_hz_to_cg_intake_bridge_pass`.
- mapped rows: `5`; ready if activated now: `0`.
- Remaining blocker: HZ confirmation fields are blank and require user-confirmed terms/source identity before guarded conversion.
- Claim boundary: no download, no conversion, no training, no evaluation, no metric/seconds-level claim, no Stage5C, no SMC.
<!-- STAGE42_IA_HZ_TO_CG_INTAKE_BRIDGE:END -->

<!-- STAGE42_IB_IA_BRIDGED_VALIDATOR_DRY_RUN:START -->
## Stage42-IB IA Bridged Validator Dry Run

- source: `fresh_stage42_ib_ia_bridged_validator_dry_run`
- role: dry-run validate the IA bridged intake using CG source-terms semantics without activating conversion.
- gate: `16 / 16`; verdict `stage42_ib_ia_bridged_validator_dry_run_pass`.
- targets validated: `5`; ready targets: `0`.
- Current result is correctly blocked because HZ user-confirmation fields remain blank.
- Claim boundary: no download, no conversion, no training, no evaluation, no metric/seconds-level claim, no Stage5C, no SMC.
<!-- STAGE42_IB_IA_BRIDGED_VALIDATOR_DRY_RUN:END -->

<!-- STAGE42_IC_CURRENT_CLAIM_EVIDENCE_CLOSURE:START -->
## Stage42-IC Current Claim / Evidence Closure

- source: `fresh_stage42_ic_current_claim_evidence_closure`
- verdict: `stage42_ic_current_claim_evidence_closure_pass`; gates `16 / 16`.
- supported claims: `6`; blocked/diagnostic claims: `7`.
- t100 row replay rows: `47458`; source terms conversion-ready now: `0`.
- IC closes the current paper-package claim map: supported claims remain protected dataset-local/raw-frame 2.5D, while true-3D/foundation/metric-seconds/Stage5C/SMC and JEPA/Transformer independent-main claims remain blocked.
- This is not new training, download, conversion, or evaluation; it is a claim/evidence closure over existing fresh/cached_verified artifacts.
<!-- STAGE42_IC_CURRENT_CLAIM_EVIDENCE_CLOSURE:END -->

<!-- STAGE42_ID_PAPER_CLAIM_CONTRACT:START -->
## Stage42-ID Paper Claim Contract

- source: `fresh_stage42_id_paper_claim_contract`
- verdict: `stage42_id_paper_claim_contract_pass`; gates `15 / 15`.
- contract rows: `13`; supported claims `6`; blocked claims `7`.
- paper files existing: `8 / 8`; files with raw/dataset caveat `8`; files with Stage5C/SMC boundary `8`.
- ID locks manuscript wording: supported claims are protected dataset-local/raw-frame 2.5D; true-3D/foundation/metric-seconds/Stage5C/SMC claims remain forbidden.
- This is a paper-claim contract only, not new training, conversion, download, or evaluation.
<!-- STAGE42_ID_PAPER_CLAIM_CONTRACT:END -->

<!-- STAGE42_IE_PAPER_CONTRACT_COMPLIANCE:START -->
## Stage42-IE Paper Contract Compliance

- source: `fresh_stage42_ie_paper_contract_compliance`
- verdict: `stage42_ie_paper_contract_compliance_pass`; gates `14 / 14`.
- paper files checked: `9 / 9`.
- supported anchors covered: `5 / 5`; unbounded overclaim hits: `0`.
- blocked claims covered as limitations: `7 / 7`.
- IE verifies the current paper package obeys the Stage42-ID contract: protected dataset-local/raw-frame 2.5D only; no true-3D/foundation/metric-seconds/Stage5C/SMC overclaim.
- This is compliance verification only, not new training, conversion, download, or evaluation.
<!-- STAGE42_IE_PAPER_CONTRACT_COMPLIANCE:END -->

<!-- PUBLIC_README_HUMAN_TONE_UPDATE:START -->
## Public GitHub README Human-Tone Update

- source: `fresh_public_readme_human_tone_update`
- Updated root `README.md` from stage-log style into a project-owner introduction for GitHub readers.
- The public README now explains M3W in first-person project language: what it is, what it currently does, what it does not claim, current trusted evidence, safety-floor role, repository map, and reproducibility entry points.
- Claim boundary preserved: protected dataset-local/raw-frame 2.5D only; not true 3D, not foundation, not metric/seconds-level, no Stage5C execution, no SMC.
- This is documentation tone cleanup only, not new training, conversion, download, or evaluation.
<!-- PUBLIC_README_HUMAN_TONE_UPDATE:END -->

<!-- STAGE42_IF_T50_GAIN_HARM_STABILITY_AUDIT:START -->
## Stage42-IF T50 Gain/Harm Stability Audit

- source: `fresh_stage42_if_t50_gain_harm_stability_audit`
- verdict: `stage42_if_t50_gain_harm_ci_blocker_identified`
- gates: `13 / 14`
- ADE t50 mean / CI low: `0.006596` / `-0.017931`
- FDE t50 mean / CI low: `0.057431` / `0.046360`
- negative ADE t50 seeds: `1`
- validation-selected seed test ADE t50: `0.028352`
- row bootstrap status: `not_run_blocked_by_missing_row_errors_in_stage42p_artifact`
- conclusion: Stage42-P is positive on mean t+50 and stable on FDE t+50, but ADE t+50 is not yet seed-CI stable enough for a paper-level t+50 ADE claim.
- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_IF_T50_GAIN_HARM_STABILITY_AUDIT:END -->

<!-- STAGE42_IG_T50_GAIN_HARM_ROW_BOOTSTRAP:START -->
## Stage42-IG T50 Gain/Harm Row Bootstrap

- source: `fresh_stage42_ig_t50_gain_harm_row_bootstrap`
- verdict: `stage42_ig_row_bootstrap_validates_selected_seed_with_multiseed_blocker`
- gates: `15 / 15`
- validation-selected seed: `151`
- selected ADE t50 / CI low: `0.028352` / `0.023371`
- selected FDE t50 / CI low: `0.067566` / `0.060976`
- selected ADE hard/failure: `0.054677`
- selected ADE easy degradation: `0.007574`
- multiseed ADE t50 CI low remains: `-0.017931`
- conclusion: validation-selected row-level t+50 evidence is positive, but seed-stable ADE t+50 remains an open blocker.
- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_IG_T50_GAIN_HARM_ROW_BOOTSTRAP:END -->

<!-- STAGE42_IH_T50_GAIN_HARM_SEED_EXPANSION:START -->
## Stage42-IH T50 Gain/Harm Seed Expansion

- source: `fresh_stage42_ih_t50_gain_harm_seed_expansion`
- verdict: `stage42_ih_t50_seed_expansion_mean_positive_ci_blocker_remains`
- gates: `15 / 16`
- combined seeds: `6`
- expanded ADE t50 mean / CI low: `0.006727` / `-0.008183`
- expanded FDE t50 mean / CI low: `0.059987` / `0.054343`
- expanded ADE hard/failure mean: `0.053456`
- expanded ADE easy degradation mean: `0.011112`
- validation-selected seed: `151` with ADE t50 `0.028352`
- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_IH_T50_GAIN_HARM_SEED_EXPANSION:END -->

<!-- STAGE42_II_T50_GAIN_HARM_ENSEMBLE_REPAIR:START -->
## Stage42-II T50 Gain/Harm Ensemble Repair

- source: `fresh_stage42_ii_t50_gain_harm_ensemble_repair`
- verdict: `stage42_ii_ensemble_repair_stabilizes_t50`
- gates: `15 / 15`
- ADE all / t50 / hard: `0.121192` / `0.081363` / `0.124775`
- ADE t50 row CI low: `0.074234`
- FDE t50 / CI low: `0.209983` / `0.202250`
- easy degradation: `0.000000`
- TrajNet t50: `0.164168`
- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_II_T50_GAIN_HARM_ENSEMBLE_REPAIR:END -->

<!-- STAGE42_IJ_T50_ENSEMBLE_SOURCE_ROBUSTNESS:START -->
## Stage42-IJ T50 Ensemble Source Robustness

- source: `fresh_stage42_ij_t50_ensemble_source_robustness`
- verdict: `stage42_ij_t50_ensemble_source_robustness_pass`
- gates: `15 / 15`
- all / t50 / hard: `0.121192` / `0.081363` / `0.124775`
- source-group t50 CI low: `0.000000`
- scene-group t50 CI low: `0.000000`
- powered t50 source positives: `2 / 3`
- easy degradation: `0.000000`
- boundary: cached-verified Stage42-II intermediates plus fresh source/scene eval; dataset-local/raw-frame 2.5D only; no metric/seconds, no Stage5C, no SMC.
<!-- STAGE42_IJ_T50_ENSEMBLE_SOURCE_ROBUSTNESS:END -->

<!-- STAGE42_IK_T50_ENSEMBLE_UCY_SPECIALIST_INTEGRATION:START -->
## Stage42-IK T50 Ensemble UCY Specialist Integration

- source: `fresh_stage42_ik_t50_ensemble_ucy_specialist_integration`
- verdict: `stage42_ik_ucy_specialist_integration_pass`
- gates: `16 / 16`
- ADE all / t50 / hard: `0.158819` / `0.104522` / `0.163730`
- ADE t50 row CI low: `0.097328`
- FDE t50 / CI low: `0.263687` / `0.256358`
- easy degradation: `0.000000`
- UCY t50: `0.122892`
- boundary: source-specialist composition evidence only; dataset-local/raw-frame 2.5D; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_IK_T50_ENSEMBLE_UCY_SPECIALIST_INTEGRATION:END -->

<!-- STAGE42_IL_T50_UCY_SPECIALIST_CLAIM_AUDIT:START -->
## Stage42-IL T50 UCY Specialist Claim Audit

- source: `fresh_stage42_il_t50_ucy_specialist_claim_audit`
- verdict: `stage42_il_ucy_specialist_claim_audit_pass`
- gates: `16 / 16`
- Stage42-IK vs Stage42-II delta all/t50/hard: `0.037627` / `0.023160` / `0.038954`
- UCY t50 before/after: `0.000000` -> `0.122892`
- non-UCY max abs metric delta: `0.000000101979`
- boundary: claim audit only; IK is source-specialist composition evidence, not independent-domain, metric/seconds, Stage5C, or SMC evidence.
<!-- STAGE42_IL_T50_UCY_SPECIALIST_CLAIM_AUDIT:END -->

<!-- STAGE42_IM_T50_SOURCE_SPECIALIST_POLICY_FREEZE:START -->
## Stage42-IM T50 Source-Specialist Policy Freeze

- source: `cached_verified_stage42_ik_il_t50_source_specialist_policy_freeze`
- verdict: `stage42_im_t50_source_specialist_policy_freeze_pass`
- gates: `22 / 22`
- policy artifact: `outputs/stage42_long_research/frozen_t50_source_specialist_policy_stage42.json`
- policy hash: `20df05705d9038ed8c3ba8b05128ca4b211aacf4b3ccbb6ea2d8f8cdb5a93ec5`
- ADE all / t50 / hard: `0.158819` / `0.104522` / `0.163730`
- FDE t50: `0.263687`
- easy degradation: `0.000000`
- boundary: frozen source-specialist t50 policy; dataset-local/raw-frame 2.5D; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_IM_T50_SOURCE_SPECIALIST_POLICY_FREEZE:END -->

<!-- STAGE42_IN_T50_SOURCE_SPECIALIST_REVIEWER_REPLAY:START -->
## Stage42-IN T50 Source-Specialist Reviewer Replay Package

- source: `cached_verified_stage42_ik_il_im_t50_source_specialist_reviewer_replay`
- verdict: `stage42_in_t50_source_specialist_reviewer_replay_pass`
- gates: `25 / 25`
- commands file: `outputs/stage42_long_research/t50_source_specialist_replay_commands_stage42.sh`
- policy hash: `20df05705d9038ed8c3ba8b05128ca4b211aacf4b3ccbb6ea2d8f8cdb5a93ec5`
- ADE all / t50 / hard: `0.158819` / `0.104522` / `0.163730`
- UCY t50 before -> after: `0.000000` -> `0.122892`
- boundary: reviewer replay package for source-specialist t50 evidence only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_IN_T50_SOURCE_SPECIALIST_REVIEWER_REPLAY:END -->

<!-- STAGE42_IO_HORIZON_SEQUENCE_GRAPH_CONTEXT_ROUTER:START -->
## Stage42-IO Horizon-Specific Sequence+Graph Context Router

- source: `fresh_stage42_horizon_sequence_graph_context_router`
- role: tests whether splitting t10/t25/t50/t100 fixes the negative Stage42-EQ global sequence+graph context router.
- gate: `13 / 13`; verdict `stage42_io_horizon_sequence_graph_context_router_pass`.
- positive_horizon_sequence_graph_context_routers: `['h10_history_only', 'h10_motion_goal_context', 'h25_baseline_plus_history_goal_neighbor']`.
- best_overall_router: `h10_motion_goal_context`.
- best all/t50/t100raw/hard/easy: `0.069270` / `0.000000` / `0.000000` / `0.072655` / `-0.035269`.
- horizon_specific_increment_verdict: `stage42_io_horizon_sequence_graph_context_router_supported`.
- Boundary: fresh horizon-specific router audit only; raw-frame/dataset-local 2.5D; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_IO_HORIZON_SEQUENCE_GRAPH_CONTEXT_ROUTER:END -->

<!-- STAGE42_IP_T50_T100_SEQUENCE_GRAPH_BLOCKER_AUDIT:START -->
## Stage42-IP t50/t100 Sequence+Graph Blocker Audit

- source: `fresh_stage42_t50_t100_sequence_graph_blocker_audit`
- role: explains why Stage42-IO sequence+graph context did not become deployable at t50/t100.
- gate: `12 / 12`; verdict `stage42_ip_t50_t100_sequence_graph_blocker_audit_pass`.
- t50_diagnosis: `router_under_switches_despite_headroom`.
- t100_diagnosis: `weak_predictive_signal_or_baseline_family_dominance`.
- blocker_counts: `{'unsafe_or_uncalibrated_switching': 2, 'weak_predictive_signal_or_baseline_family_dominance': 2, 'router_under_switches_despite_headroom': 1, 'low_margin_candidate_ambiguity': 1}`.
- conclusion: blocker audit only; no new deployable model and no t50/t100 context contribution claim.
- Boundary: raw-frame/dataset-local 2.5D; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_IP_T50_T100_SEQUENCE_GRAPH_BLOCKER_AUDIT:END -->

<!-- STAGE42_IQ_T50_SWITCHABILITY_CALIBRATION_REPAIR:START -->
## Stage42-IQ t50 Switchability Calibration Repair

- source: `fresh_stage42_t50_switchability_calibration_repair`
- role: formal repair attempt for Stage42-IP t50 under-switching using validation-selected gain/harm calibration.
- gate: `11 / 11`; verdict `stage42_iq_t50_switchability_calibration_repair_pass`.
- repair_supported: `False`; repair_verdict `validation_selected_gain_harm_router_still_fails_to_capture_t50_headroom`.
- best_trial: `baseline_plus_history_goal_neighbor__gain_only`.
- best test t50 / hard / easy: `0.000001` / `0.000001` / `-0.000000`.
- conclusion: if unsupported, do not continue pure threshold tuning; next step needs changed supervision/source support/candidate family.
- Boundary: raw-frame/dataset-local 2.5D; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_IQ_T50_SWITCHABILITY_CALIBRATION_REPAIR:END -->

<!-- STAGE42_IR_T50_SOURCE_PATTERN_SWITCHABILITY_REPAIR:START -->
## Stage42-IR t50 Source-Pattern Switchability Repair

- source: `fresh_stage42_t50_source_pattern_switchability_repair`
- role: formal source-support repair attempt for Stage42-IQ t50 switchability failure.
- gate: `11 / 11`; verdict `stage42_ir_t50_source_pattern_switchability_repair_pass`.
- repair_supported: `False`; repair_verdict `t50_source_pattern_switchability_repair_not_supported`.
- best_trial: `history_only__gain_only`.
- best test t50 / hard / easy: `0.000000` / `0.000000` / `-0.000000`.
- conclusion: source-pattern support does not repair the context t50 route under this protocol; future repair needs new candidate policies or source data.
- Boundary: raw-frame/dataset-local 2.5D; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_IR_T50_SOURCE_PATTERN_SWITCHABILITY_REPAIR:END -->

<!-- STAGE42_IS_DATA_CALIBRATION_REFRESH:START -->
## Stage42-IS Data Calibration / Source-Specific Dry-Run Refresh

- source: `fresh_run_on_current_head_after_stage42_ir`
- role: refreshes Stage42-A/BN/DW data and calibration evidence after the t50 source-pattern repair failed.
- gates: Stage42-A `7 / 7`, Stage42-BN `13 / 13`, Stage42-DW `15 / 15`.
- external ready from existing state: `opentraj, eth_ucy, trajnet, ucy`.
- source-specific calibration candidates: `ETH_seq_eth`, `ETH_seq_hotel`, `UCY_zara01`, `UCY_zara02`, `UCY_zara03`, `UCY_students03`.
- technical conversion ready after terms: `5 / 6`; estimated t50/t100 windows: `10060 / 5696`.
- source-CV after terms: `UCY` only; ETH/BIWI has too few calibrated sources, TrajNet remains short-snippet diagnostic, AerialMPT raw path remains missing.
- conclusion: next credible progress should use legal/confirmed source-specific UCY conversion or new source data; no global metric/seconds claim is allowed.
- verification: focused pytest `10 passed`; full pytest `1110 passed in 1980.35s`.
<!-- STAGE42_IS_DATA_CALIBRATION_REFRESH:END -->

<!-- STAGE42_IT_SOURCE_LEVEL_FULL_WAYPOINT_REFRESH:START -->
## Stage42-IT Source-Level Full-Waypoint Fresh Refresh

- source: `fresh_run_on_current_head`
- role: reruns Stage42-AM proposed source-level split full-waypoint evaluation after the Stage42-IS calibration refresh.
- gate: `12 / 12`; verdict `stage42_am_source_level_full_waypoint_eval_pass_positive`.
- test rows: `47458`; domains: TrajNet `37918`, UCY `9540`; full-waypoint rows: `32056`.
- protected full-waypoint ADE improvement all/t50/t100raw/hard: `0.245788` / `0.220171` / `0.143652` / `0.237494`.
- protected full-waypoint FDE improvement all/t50/t100raw/hard: `0.221325` / `0.222358` / `0.128623` / `0.213338`.
- bootstrap CI low all/t50/t100raw/hard: `0.242554` / `0.215923` / `0.137653` / `0.233887`.
- domain split: TrajNet positive; UCY remains fallback-only in this proposed source-level test.
- conclusion: full-waypoint source-level evidence remains positive on current HEAD, but still protected dataset-local/raw-frame 2.5D; no metric/seconds, no true 3D, no Stage5C, no SMC.
- verification: focused pytest `3 passed`; full pytest `.venv-pytorch/bin/python -m pytest tests -> 1110 passed in 4392.72s (1:13:12)`.
<!-- STAGE42_IT_SOURCE_LEVEL_FULL_WAYPOINT_REFRESH:END -->

<!-- STAGE42_IU_SOURCE_LEVEL_UCY_FULL_WAYPOINT_INTEGRATION:START -->
## Stage42-IU Source-Level UCY Full-Waypoint Specialist Integration

- source: `fresh_composition_from_current_stage42_it_and_cached_verified_stage42_v`
- role: closes the Stage42-IT UCY fallback-only source-level weakness by retaining Stage42-IT TrajNet and importing the cached-verified Stage42-V UCY specialist slice.
- gate: `17 / 17`; verdict `stage42_iu_source_level_ucy_full_waypoint_integration_pass`.
- rows: `47458`; domains: TrajNet + UCY.
- weighted ADE all/t50/t100raw/hard: `0.305568` / `0.284549` / `0.195280` / `0.302105`.
- weighted easy degradation: `-0.242171`.
- positive domains all/t50/t100raw/hard: `['TrajNet', 'UCY']` / `['TrajNet', 'UCY']` / `['TrajNet', 'UCY']` / `['TrajNet', 'UCY']`.
- limitation: no single merged row-cache artifact yet; this is source-level policy-package composition evidence.
- boundary: protected dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no Stage5C, no SMC.
<!-- STAGE42_IU_SOURCE_LEVEL_UCY_FULL_WAYPOINT_INTEGRATION:END -->
<!-- STAGE42_IV_SOURCE_LEVEL_ROW_CACHE_INTEGRATION:START -->
## Stage42-IV Source-Level Row-Cache Full-Waypoint Integration

- source: `fresh_run_current_source_level_row_cache_and_cached_verified_stage42v_ucy`
- role: turns the Stage42-IU TrajNet+UCY source-level policy package into a single row-level merged cache with bootstrap.
- gate: `20 / 20`; verdict `stage42_iv_source_level_row_cache_integration_pass`.
- replay: fresh on current HEAD `e043235`; focused pytest `6 passed`; full pytest `1193 passed`.
- rows: `47458`; domains: `{'TrajNet': 37918, 'UCY': 9540}`.
- ADE all/t50/t100raw/hard: `0.291543` / `0.247045` / `0.196335` / `0.287273`.
- easy degradation: `0.000000`.
- bootstrap t50 CI: `[0.242930, 0.251388]`; bootstrap_n `2000`.
- limitation: cache is local and not committed; claims remain dataset-local/raw-frame 2.5D.
- boundary: no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_IV_SOURCE_LEVEL_ROW_CACHE_INTEGRATION:END -->

<!-- STAGE42_IW_ROW_CACHE_MECHANISM_AUDIT:START -->
## Stage42-IW Source-Level Row-Cache Mechanism Audit

- source: `fresh_run_row_cache_mechanism_audit_from_cached_verified_stage42iv_cache`
- role: mechanism audit over the Stage42-IV single merged row-cache, not a new metric-only summary.
- gate: `18 / 18`; verdict `stage42_iw_row_cache_mechanism_audit_pass`.
- replay: fresh on current HEAD `e043235`; focused pytest `6 passed`; full pytest `1193 passed`.
- rows: `47458`; domain rows: `{'TrajNet': 37918, 'UCY': 9540}`.
- ADE all/t50/t100raw/hard: `0.291543` / `0.247045` / `0.196335` / `0.287273`.
- easy degradation: `0.000000`; switch rows `33355`; fallback exact floor rate `1.000000`.
- full-waypoint coverage: `0.675460`; bootstrap t50 CI `[0.242612, 0.251123]`.
- interpretation: safe-switch and teacher/floor protection are directly supported by this row-cache; waypoint labels are sequence-capable but not complete for every row; history/neighbor/goal/interaction still require retrained ablation evidence.
- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_IW_ROW_CACHE_MECHANISM_AUDIT:END -->

<!-- STAGE42_IX_SOURCE_LEVEL_CONTEXT_REPAIR:START -->
## Stage42-IX Source-Level Context Repair Trials

- source: `fresh_run_weighted_floor_residual_context_repair`
- role: retrained repair attempt after Stage42-AO showed context was not incremental after baseline-family rollout features.
- gate: `11 / 12`; verdict `stage42_ix_context_repair_completed_context_not_proven`.
- tested: `6` weighted/floor-residual variants.
- best_trial: `baseline_family_absolute_weighted`; best all/t50/t100raw/hard `0.280381` / `0.317359` / `0.143387` / `0.269583`.
- easy degradation: `-0.311860`.
- positive_context_repair_trials: `[]`.
- context_claim_verdict: `stage42_ix_context_repair_negative_context_still_not_incremental`.
- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_IX_SOURCE_LEVEL_CONTEXT_REPAIR:END -->

<!-- STAGE42_IY_SOURCE_LEVEL_NONLINEAR_CONTEXT_REPAIR:START -->
## Stage42-IY Source-Level Nonlinear Context Repair

- source: `fresh_run_sampled_extra_trees_context_capacity_repair`
- role: nonlinear capacity test after Stage42-IX still failed to make context incremental.
- gate: `12 / 13`; verdict `stage42_iy_nonlinear_context_repair_completed_context_not_proven`.
- trials: `4` ExtraTrees residual models; deterministic train cap `120000`.
- best_trial: `tree_baseline_family_residual`; best all/t50/t100raw/hard `0.221602` / `0.246937` / `0.187483` / `0.232718`.
- easy degradation: `-0.125700`.
- positive_nonlinear_context_trials: `[]`.
- capacity_hypothesis_verdict: `stage42_iy_nonlinear_context_capacity_not_sufficient`.
- boundary: sampled train-only nonlinear repair; dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_IY_SOURCE_LEVEL_NONLINEAR_CONTEXT_REPAIR:END -->

<!-- STAGE42_IZ_SOURCE_LEVEL_NONLINEAR_CONTEXT_SLICE_AUDIT:START -->
## Stage42-IZ Source-Level Nonlinear Context Slice Audit

- source: `fresh_run_retrained_extra_trees_context_slice_audit`
- role: after Stage42-IY, test whether nonlinear context has only local slice-level utility.
- gate: `11 / 11`; verdict `stage42_iz_context_slice_audit_positive`.
- supported_context_slice_count: `14`.
- decision: `context_has_powered_slice_level_support`.
- blocker_counts: `{'no_powered_positive_context_slice': 0, 'context_below_baseline_family': 55, 'easy_or_safety_not_primary_blocker': 2}`.
- boundary: train-only slice thresholds, validation-selected safe policy, test-once audit; dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_IZ_SOURCE_LEVEL_NONLINEAR_CONTEXT_SLICE_AUDIT:END -->

<!-- STAGE42_JA_CONTEXT_SLICE_POLICY_PROMOTION:START -->
## Stage42-JA Context-Slice Policy Promotion Audit

- source: `fresh_run_validation_selected_context_slice_policy`
- role: promote Stage42-IZ slice-level context evidence into a validation-selected fallback-safe policy, or reject promotion.
- gate: `10 / 12`; verdict `stage42_ja_context_slice_policy_not_promotable`.
- selected_rule_count: `13`; test_context_rule_coverage_rate `0.977327`.
- context policy all/t50/t100raw/hard/easy: `0.203253` / `0.190761` / `0.107057` / `0.195825` / `-0.211871`.
- delta vs baseline-family all/t50/t100raw/hard/easy: `-0.023421` / `-0.070733` / `-0.084708` / `-0.042885` / `-0.069684`.
- decision: `validation_selected_context_slice_policy_not_promoted`.
- boundary: validation-only slice policy selection, test-once evaluation; dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_JA_CONTEXT_SLICE_POLICY_PROMOTION:END -->

<!-- STAGE42_JB_CONSERVATIVE_CONTEXT_SLICE_POLICY_REPAIR:START -->
## Stage42-JB Conservative Context-Slice Policy Repair

- source: `fresh_run_validation_greedy_conservative_context_slice_repair`
- role: after Stage42-JA failed, try a stricter validation-greedy, inference-safe, core-preserving context slice repair.
- gate: `11 / 13`; verdict `stage42_jb_conservative_context_policy_not_promotable`.
- selected_rule_count: `4`; test_context_rule_coverage_rate `0.526950`.
- conservative policy all/t50/t100raw/hard/easy: `0.231382` / `0.190761` / `0.191765` / `0.227164` / `-0.220374`.
- delta vs baseline-family all/t50/t100raw/hard/easy: `0.004708` / `-0.070733` / `0.000000` / `-0.011546` / `-0.078187`.
- primary_blocker: `context_policy_has_core_metric_regression`.
- boundary: validation-greedy policy selection, test-once evaluation; dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_JB_CONSERVATIVE_CONTEXT_SLICE_POLICY_REPAIR:END -->

<!-- STAGE42_JC_LATEST_EVIDENCE_TIER_CONSOLIDATION:START -->
## Stage42-JC Latest Evidence Tier Consolidation

- source: `fresh_stage42_jc_latest_evidence_tier_consolidation`
- gate: `20 / 20`; verdict: `stage42_jc_latest_evidence_tier_consolidation_pass`
- main evidence: `T1_source_level_row_cache_full_waypoint` with all `29.15%`, t50 `24.70%`, t100 raw-frame diagnostic `19.63%`, hard/failure `28.73%`, easy degradation `0.00%`.
- context boundary: Stage42-IZ has `14` local supported context slices, but JA/JB failed promotion, so context is not a deployable/global main contribution.
- claim boundary: still protected dataset-local/raw-frame 2.5D; not true 3D, not foundation, not metric/seconds-level, no Stage5C, no SMC.
<!-- STAGE42_JC_LATEST_EVIDENCE_TIER_CONSOLIDATION:END -->

<!-- STAGE42_JD_CALIBRATION_READINESS_RECONCILIATION:START -->
## Stage42-JD Calibration Readiness Reconciliation

- source: `fresh_stage42_jd_calibration_readiness_reconciliation`
- gate: `21 / 21`; verdict: `stage42_jd_calibration_readiness_reconciliation_pass`
- required datasets covered: `['aerialmpt', 'eth_ucy', 'opentraj', 'sdd', 'tgsim', 'trajnet', 'ucy']`; direct path groups found `9 / 9`.
- source-specific metric/time candidates: `7`; ready now: `False`.
- conclusion: external validation/full-waypoint work can continue in raw-frame/dataset-local mode, but metric/seconds claims remain blocked until user-confirmed terms, guarded conversion, no-leakage, and restricted evaluation.
- Stage5C not executed; SMC not enabled.
<!-- STAGE42_JD_CALIBRATION_READINESS_RECONCILIATION:END -->

<!-- STAGE42_JE_SOURCE_ROTATION_FULL_WAYPOINT_EVAL:START -->
## Stage42-JE Source-Rotation Full-Waypoint Evaluation

- source: `fresh_stage42_je_source_rotation_full_waypoint_eval`
- gate: `14 / 14`; verdict: `stage42_je_source_rotation_full_waypoint_eval_pass`
- held-out domain rotations: ETH_UCY: all 25.23%, t50 21.07%, hard 26.08%, easy 27.83%; TrajNet: all 30.11%, t50 39.29%, hard 29.21%, easy -24.27%; UCY: all 21.86%, t50 23.73%, hard 20.19%, easy -21.09%.
- decision: `source_rotation_positive_but_not_global_deployable`; deployable held-out domains: `['TrajNet', 'UCY']`.
- boundary: this is stricter cross-domain raw-frame evidence; it does not change the no-metric/no-seconds/no-Stage5C/no-SMC boundary.
<!-- STAGE42_JE_SOURCE_ROTATION_FULL_WAYPOINT_EVAL:END -->

<!-- STAGE42_JF_SOURCE_ROTATION_EASY_GUARD_REPAIR:START -->
## Stage42-JF Source-Rotation Easy-Guard Repair

- source: `fresh_stage42_jf_source_rotation_easy_guard_repair`
- gate: `9 / 9`; verdict: `stage42_jf_source_rotation_easy_guard_repair_pass`
- held-out easy-guard rotations: ETH_UCY: cap 1.00, all 25.23%, t50 21.07%, hard 26.08%, easy 27.83%; TrajNet: cap 0.75, all 30.13%, t50 39.29%, hard 29.19%, easy -25.02%; UCY: cap 0.75, all 21.86%, t50 23.73%, hard 20.19%, easy -21.09%.
- decision: `easy_guard_repair_partial_domain_bounded`; deployable domains after easy guard: `['TrajNet', 'UCY']`; still blocked: `['ETH_UCY']`.
- boundary: validation-only switch budget; no test threshold tuning, no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JF_SOURCE_ROTATION_EASY_GUARD_REPAIR:END -->

<!-- STAGE42_JG_ETH_UCY_SOURCE_SPECIFIC_EASY_GUARD:START -->
## Stage42-JG ETH_UCY Source-Specific Easy-Guard Feasibility

- source: `fresh_stage42_jg_eth_ucy_source_specific_easy_guard`
- gate: `11 / 11`; verdict: `stage42_jg_eth_ucy_source_specific_easy_guard_pass`
- source-CV folds: ETH/seq_eth/obsmat.txt: all 0.58%, t50 -32.47%, hard 0.63%, easy -11.79%; ETH/seq_hotel/obsmat.txt: all 8.64%, t50 15.05%, hard 8.70%, easy -15.89%; UCY/students03/obsmat.txt: all 8.73%, t50 9.39%, hard 10.24%, easy 19.42%; UCY/zara01/obsmat.txt: all 12.50%, t50 17.97%, hard 11.43%, easy -24.69%; UCY/zara02/obsmat.txt: all 27.54%, t50 36.18%, hard 28.92%, easy 81.62%.
- decision: `eth_ucy_source_specific_policy_partial_source_support`; deployable sources: `['ETH/seq_hotel/obsmat.txt', 'UCY/zara01/obsmat.txt']`; blocked sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt', 'UCY/zara02/obsmat.txt']`.
- boundary: this is ETH_UCY source-specific support only, not cross-domain zero-shot success; still no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JG_ETH_UCY_SOURCE_SPECIFIC_EASY_GUARD:END -->

<!-- STAGE42_JH_ETH_UCY_HARM_AWARE_SOURCE_GUARD:START -->
## Stage42-JH ETH_UCY Harm-Aware Source Guard

- source: `fresh_stage42_jh_eth_ucy_harm_aware_source_guard`
- gate: `9 / 9`; verdict: `stage42_jh_eth_ucy_harm_aware_source_guard_pass`
- source-CV harm-aware folds: ETH/seq_eth/obsmat.txt: all 0.58%, t50 -32.47%, hard 0.63%, easy -11.82%; ETH/seq_hotel/obsmat.txt: all 8.64%, t50 15.05%, hard 8.70%, easy -15.89%; UCY/students03/obsmat.txt: all 9.09%, t50 9.03%, hard 10.02%, easy 10.78%; UCY/zara01/obsmat.txt: all 12.50%, t50 17.97%, hard 11.43%, easy -24.69%; UCY/zara02/obsmat.txt: all 30.39%, t50 38.99%, hard 30.27%, easy -2.52%.
- decision: `eth_ucy_harm_aware_guard_partial_support`; deployable sources: `['ETH/seq_hotel/obsmat.txt', 'UCY/zara01/obsmat.txt', 'UCY/zara02/obsmat.txt']`; blocked sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`; easy repaired: `['UCY/zara02/obsmat.txt']`.
- boundary: this is ETH_UCY source-specific support only, not global/cross-domain success; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JH_ETH_UCY_HARM_AWARE_SOURCE_GUARD:END -->

<!-- STAGE42_JI_ETH_UCY_SOURCE_ROBUST_BLOCKED_REPAIR:START -->
## Stage42-JI ETH_UCY Source-Robust Blocked-Source Repair

- source: `fresh_stage42_ji_eth_ucy_source_robust_blocked_repair`
- gate: `10 / 10`; verdict: `stage42_ji_eth_ucy_source_robust_blocked_repair_pass`
- targets from JH blocked sources: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`
- repair folds: ETH/seq_eth/obsmat.txt: all 0.97%, t50 -31.92%, hard 1.05%, easy -14.48%, deployable=False; UCY/students03/obsmat.txt: all 5.42%, t50 3.69%, hard 6.23%, easy 7.24%, deployable=False.
- decision: `eth_ucy_blocked_sources_still_blocked`; repaired: `[]`; still blocked: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`; easy improved: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`.
- boundary: held-out sources still blocked remain fallback-only; no global ETH_UCY/cross-domain overclaim, no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JI_ETH_UCY_SOURCE_ROBUST_BLOCKED_REPAIR:END -->

<!-- STAGE42_JJ_ETH_UCY_BLOCKED_SOURCE_GEOMETRY_SUPPORT:START -->
## Stage42-JJ ETH_UCY Blocked-Source Geometry/Family Support

- source: `fresh_stage42_jj_eth_ucy_blocked_source_geometry_support`
- gate: `11 / 11`; verdict: `stage42_jj_eth_ucy_blocked_source_geometry_support_pass`
- family/geometry support audit: ETH/seq_eth/obsmat.txt: static all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, family-oracle t50 53.80%, deployable=False; UCY/students03/obsmat.txt: static all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, family-oracle t50 39.14%, deployable=False.
- decision: `blocked_sources_not_repaired_family_support_diagnostic`; repaired: `[]`; still blocked: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`.
- boundary: static causal family support does not globally repair ETH_UCY; blocked sources stay fallback-only; no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JJ_ETH_UCY_BLOCKED_SOURCE_GEOMETRY_SUPPORT:END -->

<!-- STAGE42_JK_ETH_UCY_ROW_FAMILY_SELECTOR:START -->
## Stage42-JK ETH_UCY Row-Level Family Selector

- source: `fresh_stage42_jk_eth_ucy_row_family_selector`
- gate: `11 / 11`; verdict: `stage42_jk_eth_ucy_row_family_selector_pass`
- row-family heldout results: ETH/seq_eth/obsmat.txt: all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, oracle t50 53.80%, deployable=False; UCY/students03/obsmat.txt: all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, oracle t50 39.14%, deployable=False.
- decision: `row_family_selector_not_deployable_on_blocked_sources`; repaired: `[]`; still blocked: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`.
- boundary: no full ETH_UCY/cross-domain overclaim; still dataset-local raw-frame 2.5D, no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JK_ETH_UCY_ROW_FAMILY_SELECTOR:END -->

<!-- STAGE42_JL_ETH_UCY_SOURCE_SUPPORT_COVERAGE:START -->
## Stage42-JL ETH_UCY Source Support Coverage

- source: `fresh_stage42_jl_eth_ucy_source_support_coverage`
- gate: `11 / 11`; verdict: `stage42_jl_eth_ucy_source_support_coverage_pass`
- source-support heldout results: ETH/seq_eth/obsmat.txt: support=True, all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, oracle t50 53.80%; UCY/students03/obsmat.txt: support=False, all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, oracle t50 39.14%.
- decision: `source_support_policy_not_deployable_support_blocker`; repaired: `[]`; still blocked: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`; unsupported: `['UCY/students03/obsmat.txt']`.
- boundary: this is a source-support diagnostic/repair attempt, still dataset-local raw-frame 2.5D, no metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JL_ETH_UCY_SOURCE_SUPPORT_COVERAGE:END -->

<!-- STAGE42_JM_ETH_UCY_CALIBRATED_SUPPORT_RECHECK:START -->
## Stage42-JM ETH_UCY Calibrated Support Recheck

- source: `fresh_stage42_jm_eth_ucy_calibrated_support_recheck`
- gate: `11 / 11`; verdict: `stage42_jm_eth_ucy_calibrated_support_recheck_pass`
- calibrated-support heldout results: ETH/seq_eth/obsmat.txt: local_calib=source_specific_annotation_step_meter_coordinate_evidence, all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, deployable=False; UCY/students03/obsmat.txt: local_calib=source_specific_annotation_step_meter_coordinate_evidence, all 0.00%, t50 0.00%, hard 0.00%, easy -0.00%, deployable=False.
- decision: `calibrated_support_recheck_blocked_no_safe_deployment`; repaired: `[]`; still blocked: `['ETH/seq_eth/obsmat.txt', 'UCY/students03/obsmat.txt']`.
- boundary: source-specific calibration evidence is recorded, but the main claim remains dataset-local/raw-frame 2.5D; no global metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JM_ETH_UCY_CALIBRATED_SUPPORT_RECHECK:END -->

<!-- STAGE42_JN_LOCAL_CALIBRATED_SOURCE_SUPPORT_INTAKE:START -->
## Stage42-JN Local Calibrated Source Support Intake

- source: `fresh_stage42_jn_local_calibrated_source_support_intake`
- gate: `12 / 12`; verdict: `stage42_jn_local_calibrated_source_support_intake_pass`
- parseable support candidates: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`; long-horizon candidates: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`.
- decision: `candidate_sources_found_but_user_terms_required`; auto_convert_allowed: `[]`.
- boundary: candidate-source intake only; no conversion, no deployment claim, no global metric/seconds claim, no Stage5C, no SMC.
<!-- STAGE42_JN_LOCAL_CALIBRATED_SOURCE_SUPPORT_INTAKE:END -->

<!-- STAGE42_JO_LOCAL_CALIBRATED_SOURCE_GUARDED_CONVERSION_PREFLIGHT:START -->
## Stage42-JO Local Calibrated Source Guarded Conversion Preflight

- source: `fresh_stage42_jo_local_calibrated_source_guarded_conversion_preflight`
- gate: `13 / 13`; verdict: `stage42_jo_local_calibrated_source_guarded_preflight_pass`
- technical_ready_after_terms: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`; conversion_allowed_now: `[]`.
- decision: `guarded_conversion_preflight_blocked_pending_user_terms`; blocked_by_terms: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`.
- boundary: preflight only; no conversion, no deployable source-support claim, no metric/seconds overclaim, no Stage5C, no SMC.
<!-- STAGE42_JO_LOCAL_CALIBRATED_SOURCE_GUARDED_CONVERSION_PREFLIGHT:END -->

<!-- STAGE42_JP_LOCAL_CALIBRATED_SOURCE_TERMS_PREFILL:START -->
## Stage42-JP Local Calibrated Source Terms Prefill

- source: `fresh_stage42_jp_local_calibrated_source_terms_prefill`
- gate: `15 / 15`; verdict: `stage42_jp_local_calibrated_source_terms_prefill_pass`
- official_hint_rows: `3`; license_found_rows: `1`; conversion_ready_now: `0`.
- high_confidence_official_source_rows: `['Wild-Track']`; manual_only_rows: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`.
- boundary: terms prefill only; no permission, no conversion, no evaluation, no metric/seconds overclaim, no Stage5C, no SMC.
<!-- STAGE42_JP_LOCAL_CALIBRATED_SOURCE_TERMS_PREFILL:END -->

<!-- STAGE42_JQ_LOCAL_CALIBRATED_SOURCE_TERMS_VALIDATION:START -->
## Stage42-JQ Local Calibrated Source Terms Validation

- source: `fresh_stage42_jq_local_calibrated_source_terms_validator`
- gate: `14 / 14`; verdict: `stage42_jq_local_calibrated_source_terms_validation_pass`
- datasets_validated: `3`; terms_accepted_rows: `0`; conversion_ready_rows: `0`.
- blocked_rows: `['Town-Center', 'Wild-Track', 'PETS-2009-S2L1']`; ready_for_future_guarded_conversion: `[]`.
- boundary: user terms validator only; no download, no conversion, no evaluation, no metric/seconds overclaim, no Stage5C, no SMC.
<!-- STAGE42_JQ_LOCAL_CALIBRATED_SOURCE_TERMS_VALIDATION:END -->

<!-- STAGE42_JR_SOURCE_CONTEXT_FRESH_REPLAY:START -->
## Stage42-JR Source Context Fresh Replay

- source: `fresh_stage42_jr_source_context_fresh_replay`
- gate: `12 / 12`; verdict: `stage42_jr_source_context_negative_evidence_pass`
- baseline-family all/t50/hard remains positive: `0.2878` / `0.3154` / `0.2758`.
- sequence context did not add lift: best all/t50/hard delta `-0.0245` / `-0.0831` / `-0.0284`.
- graph context did not add lift: best all/t50/hard delta `-0.0230` / `-0.0858` / `-0.0262`.
- boundary: negative result preserved; no sequence/graph independent main claim, no metric/seconds overclaim, no Stage5C, no SMC.
<!-- STAGE42_JR_SOURCE_CONTEXT_FRESH_REPLAY:END -->

<!-- STAGE42_JS_SOURCE_CONTEXT_GAIN_HARM_CLOSURE:START -->
## Stage42-JS Source Context Gain/Harm Closure

- source: `fresh_stage42_js_source_context_gain_harm_closure`
- gate: `14 / 14`; verdict: `stage42_js_source_context_gain_harm_closure_pass`
- narrow horizon positives: `['h10_history_only', 'h10_motion_goal_context', 'h25_baseline_plus_history_goal_neighbor']`; these are not t50/t100 main-claim evidence.
- t50 blocker: `router_under_switches_despite_headroom` with oracle headroom `0.0352`; IQ repair t50 `0.000001`, IR repair t50 `0.000000`.
- t100 blocker: `weak_predictive_signal_or_baseline_family_dominance` with oracle headroom `0.0112`.
- decision: close the current source-level sequence/graph gain-harm candidate family for t50/t100 independent contribution; next work needs new candidate policies or row/source-slice objectives.
- boundary: raw-frame/dataset-local 2.5D only; no metric/seconds overclaim, no Stage5C, no SMC.
<!-- STAGE42_JS_SOURCE_CONTEXT_GAIN_HARM_CLOSURE:END -->

<!-- STAGE42_JT_CURRENT_MODULE_CLAIM_REFRESH:START -->
## Stage42-JT Current Module Claim Refresh

- source: `fresh_stage42_jt_current_module_claim_refresh`
- gate: `15 / 15`; verdict: `stage42_jt_current_module_claim_refresh_pass`
- row-cache ADE all/t50/t100raw/hard: `0.291543` / `0.247045` / `0.196335` / `0.287273`; easy `0.000000`.
- AO standalone context variants: `['history_only', 'motion_goal_context']`; incremental after baseline-family: `[]`.
- blocked independent claims: `['incremental_context_after_baseline_family', 'scene_goal_independent_main_claim', 'neighbor_interaction_independent_main_claim', 'sequence_graph_t50_t100_independent_main_claim', 'JEPA_downstream_main_claim', 'Transformer_independent_main_claim', 'ungated_full_waypoint_deployment', 'metric_seconds_or_true3d_claim']`.
- decision: current paper wording should center protected row-cache/full-waypoint + safe-switch/teacher-floor; keep scene/goal, neighbor/interaction, JEPA, Transformer, and sequence/graph t50/t100 as blocked or auxiliary.
- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.
<!-- STAGE42_JT_CURRENT_MODULE_CLAIM_REFRESH:END -->
