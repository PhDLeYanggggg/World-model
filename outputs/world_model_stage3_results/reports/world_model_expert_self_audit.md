# World Model Expert Self-Audit

## Verdict

- Score: `58.0/100`
- Verdict: `prototype_with_major_failures`
- Model type: `pseudo-3D physics-informed learned residual state-space world model`
- Gates passed: `4/10`

## Blunt Expert Assessment

当前系统是有价值的研究原型，但离世界级模型还差一截。它有可验证 synthetic t+100，也有 learned residual，但多分支概率和真实数据闭环仍然薄弱。

## Gate Results

| Gate | Pass | Score | Evidence | Required Fix |
| --- | --- | --- | --- | --- |
| synthetic_t100_protocol | True | 10.0 | horizon_100_present=True; report_states_verified_synthetic_t100=True | Keep synthetic t+100 as verified evaluation and never mix it with real-data free-run metrics. |
| no_real_t100_overclaim | True | 10.0 | free_run_warning=True; suspicious_real_t100_metric=False | Keep AerialMPT t+100 qualitative-only until a real long sequence is loaded. |
| long_horizon_accuracy | False | 6.0 | hybrid_SMC ADE@100=4.216m; FDE@100=7.766m; target ADE<2m and FDE<5m | Train on larger/diverse synthetic data, add explicit goal inference, and tune model against rollout loss not only one-step residual. |
| learned_dynamics_beats_hand_physics | False | 5.0 | deterministic_gain=0.03%; hybrid_SMC_gain=-38.25%; required >=10% | Use multi-step rollout loss, residual dropout, event-balanced minibatches, and learned goal posterior instead of only one-step residual acceleration. |
| stochastic_multibranch_coverage | False | 4.0 | branch_count=16.0; coverage@64=0.25; best_FDE@100=7.005m | Make SMC particles represent latent goals/intents, report actual best-of-N, and run true 64-particle evaluation after runtime stabilizes. |
| physical_consistency | True | 12.0 | collision=0.0; boundary=0.00835; obstacle=0.0; min_gap=0.019m | Make projection cost visible in weights, add stronger wall-aware path planning, and penalize boundary drift in rollout not just one-step training. |
| constraints_reduce_violations | True | 8.0 | collision_reduction=0.50990; boundary_reduction=0.15996 | Keep physical projection, but separate correction from likelihood so invalid proposals are not silently sanitized. |
| terminal_semantic_diversity | False | 3.0 | cluster_diversity=0.74448; semantic_event_accuracy=0.0 | Cluster over intent, density, pass time, split/merge, detour, and jam features; evaluate event-balanced episodes and avoid endpoint-only semantics. |
| smc_adds_predictive_value | False | 0.0 | det_FDE@100=7.021; hybrid_best_FDE@100=7.855; learned_SMC_FDE@100=14.340; hand_FDE@100=6.830 | Use learned proposal as residual around physics, not standalone acceleration; add observation/state likelihood on synthetic and latent-goal rejuvenation. |
| real_world_readiness | False | 0.0 | real_data_limits_acknowledged=True; next_data_named=True; no calibrated long real-data loader yet | Build a long real-data loader, calibrated scene geometry, and verified t+100 real benchmark before claiming real-world readiness. |

## Priority Actions

1. Train on larger/diverse synthetic data, add explicit goal inference, and tune model against rollout loss not only one-step residual.
2. Use multi-step rollout loss, residual dropout, event-balanced minibatches, and learned goal posterior instead of only one-step residual acceleration.
3. Make SMC particles represent latent goals/intents, report actual best-of-N, and run true 64-particle evaluation after runtime stabilizes.
4. Build a long real-data loader, calibrated scene geometry, and verified t+100 real benchmark before claiming real-world readiness.
5. Cluster over intent, density, pass time, split/merge, detour, and jam features; evaluate event-balanced episodes and avoid endpoint-only semantics.
6. Use learned proposal as residual around physics, not standalone acceleration; add observation/state likelihood on synthetic and latent-goal rejuvenation.

## Bar For A Truly Exceptional World Model

- Verified t+100 on synthetic and at least one calibrated real long-trajectory dataset.
- Learned dynamics must beat hand physics by a meaningful margin, not a rounding error.
- SMC/multi-branch futures must increase coverage and produce semantically distinct terminal modes.
- Physical constraints must reduce collision, obstacle, boundary, speed, and acceleration violations without hiding bad proposals.
- Camera / homography / scene geometry uncertainty must be explicit.
- Reports must separate verified forecasts from qualitative free-run every time.
