# Physical World Model 2.5D Results

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

