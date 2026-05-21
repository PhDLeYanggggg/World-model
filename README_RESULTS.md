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
