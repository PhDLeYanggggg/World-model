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
