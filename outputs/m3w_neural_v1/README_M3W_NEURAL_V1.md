# M3W-Neural v1

M3W-Neural v1 is a Stage41-protected neural world-dynamics candidate. It combines composite-tail bounded neural dynamics with a validation-selected safe-switch policy and the Stage37/teacher safety floor.

It is not true 3D, not metric, not seconds-level, not a foundation model, and not Stage5C/SMC.

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
