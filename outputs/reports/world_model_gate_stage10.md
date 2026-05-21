# Stage 10 Gates

Passed: 5 / 10

| gate | status | pass | evidence | next fix |
| --- | --- | --- | --- | --- |
| Pedestrian/Drone Data Gate | pass | True | loaded=['trajnet', 'eth_ucy'] | Load at least one real pedestrian/drone source. |
| Long-Horizon Gate | fail | False | verified_t50_or_t100=[] | Cannot claim pedestrian long-horizon world model until this passes. |
| Human/Silver Annotation Gate | partial | False | human_confirmed=0, silver_rule_confirmed=20 | Need at least 3 gold_human or silver_human_confirmed scenes. |
| Scene Pack Gate | pass | True | walkable=27, goals=27 | Need usable walkable + goal scene packs. |
| Multi-Agent Episode Gate | partial | False | >=2_agent_episodes=320 | Need 500 multi-agent episodes or mark partial. |
| Hard/Failure Episode Gate | pass | True | hard_plus_failure=546 | Need at least 100 hard/failure episodes. |
| GoalBench v3 Gate | pass | True | official_records=1530, inferred_only_records=0 | Need 500 official non-inferred records. |
| No Leakage Gate | pass | True | train-only candidate goals; no future endpoint input; causal velocity | Repair leakage flags. |
| Stage 11 Readiness Gate | fail | False | not_ready | Pass data, human annotation, scene pack, GoalBench and no-leakage gates. |
| Stage 5C Readiness Gate | fail | False | Stage 10 is data/annotation only; latent generative remains forbidden. | Keep disabled. |

stage11_ready: `False`
latent_stage5c_ready: `False`
smc_ready: `False`
expert_audit_score: `78`
verdict: `stage10_data_annotation_package_partial_not_stage11_ready`
