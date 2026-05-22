# Stage 14 Gates

Passed: 7 / 11

| gate | pass | evidence |
| --- | --- | --- |
| Continuous Execution Gate | True | elapsed_hours=0.25; training_trials=10 |
| Multimodal Data Gate | True | At least one local pedestrian/drone source has trajectories plus scene context. |
| Long-Horizon Gate | True | per_agent_t100_rows=64 |
| Scene Pack Gate | True | scene_pack_count=43 |
| Strong Baseline Gate | True | Benchmark rows compare model FDE against baseline_FDE. |
| Deterministic Improvement Gate | False | t100=0.008052; hard=0.000000; failure=0.000000 |
| Scene/Visual Gain Gate | False | visual_gain=0.0 |
| Easy Preservation Gate | True | easy_improvement=0.038961 |
| Physical Validity Gate | True | Deterministic residuals are bounded; no SMC/stochastic rollout enabled. |
| Stage 5C Readiness Gate | False | Generate plan only if true; never execute in Stage14. |
| SMC Readiness Gate | False | No stochastic proposal or coverage gate was run. |

Do not enter Stage 5C. Deterministic multimodal correction is not strong enough.

SMC remains disabled in Stage 14.
