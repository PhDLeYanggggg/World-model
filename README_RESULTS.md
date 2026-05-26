# Physical World Model 2.5D Results

## 中文详细目标总结

Latest Chinese detailed summary of the M3W goal:

`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/README_M3W_GOAL_DETAILED_SUMMARY_ZH.md`

Full long-form Chinese route/failure/success summary for the current goal:

`/Users/yangyue/Downloads/World/outputs/m3w_neural_v1/README_M3W_LONG_GOAL_SUMMARY_ZH.md`

It records what was tried, what failed, why it failed, what worked, and the current claim boundary, now including Stage42-A through Stage42-P. The current best deployable candidate remains M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under the Stage37/teacher floor. It is still not true 3D, not metric/seconds-level, not a foundation model, and Stage5C/SMC remain disabled. Stage42-F produced a strong protected 2.5D manuscript evidence package; Stage42-G/H added fresh ablation and sequence-history evidence; Stage42-I/J/K/L/M/N/O/P investigated full-waypoint static/context repair and explicit gain/harm selector repair. Stage42-P repairs the mean ADE t+50 sign with train-only normalization and validation-only policy selection, but its 3-seed t50 CI low is still negative, so it is a gate-passing repair rather than a paper-stable t+50 claim.

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
