# M3W-Neural v1

M3W-Neural v1 is a Stage41-protected neural world-dynamics candidate. It combines composite-tail bounded neural dynamics with a validation-selected safe-switch policy and the Stage37/teacher safety floor.

It is not true 3D, not metric, not seconds-level, not a foundation model, and not Stage5C/SMC.

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

The Stage42-AE row-cache stress audit is now packaged too. It confirms Stage42-X global t50 remains seed/bootstrap positive, but it records limitations rather than overclaiming: ETH_UCY has weak t50/FDE@50 lower bounds and horizon=25 is not uniformly positive. Stage42-AF then repairs the horizon=25 weak slice with a validation-only low-margin guard while preserving the ETH_UCY t50/FDE@50 limitation.

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
