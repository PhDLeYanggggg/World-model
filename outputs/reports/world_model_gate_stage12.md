# Stage 12 Gates

Passed: 9 / 10

| gate | status | pass | evidence | next fix |
| --- | --- | --- | --- | --- |
| Pedestrian/Drone Data Gate | pass | True | loaded=['eth_ucy_ewap', 'aerialmpt', 'full_trajnet_original_quick'] | Load at least one real pedestrian/drone source. |
| Long-Horizon Gate | pass | True | verified_t50_or_t100=['eth_ucy_ewap'] | Cannot claim pedestrian long-horizon world model until this passes. |
| Human/Silver Annotation Gate | pass | True | human_confirmed=3, silver_rule_confirmed=33 | Need at least 3 gold_human or silver_human_confirmed scenes. |
| Scene Pack Gate | pass | True | walkable=43, goals=43 | Need usable scene packs. |
| Multi-Agent Episode Gate | pass | True | >=2_agent_episodes=660 | Need 500 multi-agent episodes or mark partial. |
| Hard/Failure Episode Gate | pass | True | hard_or_failure=649 | Need at least 100 hard/failure episodes. |
| GoalBench v4 Gate | pass | True | official_records=5574 | Need 500 official records. |
| No Leakage Gate | pass | True | candidate goals train-only; no future endpoint input; causal velocity | Repair leakage. |
| Stage 13 Readiness Gate | pass | True | ready | Pass Stage 12 data/annotation/GoalBench/no-leakage gates. |
| Stage 5C Readiness Gate | fail | False | Stage 12 is data/annotation only; latent generative remains forbidden. | Keep disabled. |

stage13_ready: `True`
latent_stage5c_ready: `False`
smc_ready: `False`
expert_audit_score: `83`
verdict: `stage12_ready_for_stage13_training_with_long_horizon_source`
