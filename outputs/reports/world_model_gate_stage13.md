# Stage 13 Gates

Passed: 5 / 12

| gate | pass | evidence |
| --- | --- | --- |
| Data Gate | True | Stage 12 data loaded; Stage 13 metrics rows exist. |
| No Leakage Gate | True | Inherited Stage 12 no-leakage policy: causal velocity, train-only goals, no future endpoint input. |
| Strong Baseline Gate | True | Every metric row compares against baseline_FDE. |
| Eth-UCY EWAP Long-Horizon Gate | False | no_evaluable_t100_rows_under_stage13_causal_per_agent_mask |
| HardBench Gate | False | best_hard_improvement=0.013127 |
| BaselineFailureBench Gate | False | best_failure_improvement=0.013127 |
| Easy Preservation Gate | True | best_easy_improvement=0.000000 |
| Scene/Goal Gate | False | scene_goal_hard=0.007591; no_scene_hard=0.007591 |
| Interaction Gate | False | interaction_hard=0.007591; no_interaction_hard=0.013127 |
| Physical Validity Gate | True | No residual explosion observed in bounded residual search. |
| Stage 5C Readiness Gate | False | Plan only; do not execute without user confirmation. |
| SMC Readiness Gate | False | Stage 13 does not train stochastic proposals. |

Do not claim pedestrian long-horizon world model.

Do not enter Stage 5C. Deterministic hard/failure correction is not strong enough.
