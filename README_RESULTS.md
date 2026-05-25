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
composite_tail_evidence_pass = True
composite_tail_multiseed_pass = True
all_agent_composite_world_state_pass = True
all_agent_composite_ade_all_improvement = 0.2102513255185352
all_agent_composite_ade_t50_improvement = 0.13652231450154184
all_agent_composite_fde_all_improvement = 0.19816955620782206
all_agent_composite_fde_t50_improvement = 0.17387876491801468
deployment_state = composite_tail_candidate_pending_final_package_acceptance
```

Current best candidate: M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under the Stage37/teacher safety floor. Stage37 remains the explicit fallback floor, and ungated/full-row neural dynamics are not claimed safe.
<!-- M3W_NEURAL_V1:END -->

<!-- M3W_NEURAL_COMPLETION_AUDIT:START -->
## M3W-Neural v1 Completion Audit

The active breakthrough objective is not fully complete yet. M3W-Neural v1 now has a no-base-switch joint policy distiller with bootstrap/multi-seed stability, a train-only UCY fallback repair, a grouped all-agent rollout consistency audit, a neural group-consistency distiller, a teacher-guided neural proposal repaired by a validation-selected safety guard, and a domain-local endpoint retrain checked by an endpoint-linear all-agent safety proxy on two domains. The rollout is still not a latent generative world state.

```text
completion_status = not_complete
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
strict_pure_ucy_only_neural_retrain_gate = False
domain_local_endpoint_two_domain_gate = True
domain_local_endpoint_positive_domains = ['ETH_UCY', 'TrajNet', 'UCY_expanded']
domain_local_all_agent_two_domain_gate = True
domain_local_all_agent_positive_domains = ['ETH_UCY', 'UCY_expanded']
domain_local_full_waypoint_two_domain_gate = False
domain_local_full_waypoint_positive_domains = []
domain_local_full_waypoint_failure_taxonomy = {'ETH_UCY': {'reasons': ['t50_ade_not_positive', 't100_ade_not_positive', 'same_frame_proximity_delta_unsafe', 'ungated_neural_catastrophic_easy_degradation'], 'ade_all': 0.03200185360974939, 'ade_t50': 0.0, 'ade_t100': 0.0, 'fde_all': 0.02140015213054336, 'fde_t50': 0.0, 'collision_delta_vs_floor_005': 0.020563149900689304, 'next_fix': 'train a validation-selected efficient proximity guard and horizon-specific waypoint policy; do not deploy learned full-waypoint domain-local model yet.'}, 'TrajNet': {'reasons': ['all_ade_not_positive', 't50_ade_not_positive', 'hard_failure_ade_not_positive', 'same_frame_proximity_delta_unsafe', 'ungated_neural_catastrophic_easy_degradation', 'endpoint_fde_positive_but_waypoint_ade_negative'], 'ade_all': -0.009876103097439914, 'ade_t50': -0.09471803973274517, 'ade_t100': 0.022840260699697912, 'fde_all': 0.06300366873346075, 'fde_t50': 0.06502418472340132, 'collision_delta_vs_floor_005': 0.014909478168264101, 'next_fix': 'train a validation-selected efficient proximity guard and horizon-specific waypoint policy; do not deploy learned full-waypoint domain-local model yet.'}}
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

Next target: pursue stricter pure UCY-only retrain/select/test evidence and safer no-fallback/full-row neural dynamics. Current claims remain dataset-local raw-frame 2.5D, not true 3D or foundation.
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

Composite-tail is now bootstrap-supported, multiseed-supported, pure-UCY source-heldout supported, and backed by a UCY-only policy-head train/val/test calibration. It is still not a strict pure-UCY neural retrain because the proposal/floor features remain mixed-external trained.

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
