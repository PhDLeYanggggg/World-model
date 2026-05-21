# Stage 8.5 Gates

Passed: 6 / 7

| gate | pass | evidence | next fix |
| --- | --- | --- | --- |
| Pedestrian/Drone Data Gate | True | loaded pedestrian/drone sources=['trajnet', 'eth_ucy'] | Provide local SDD/OpenTraj if this is empty. |
| Long-Horizon Gate | False | verified t50/t100 pedestrian/drone sources=[] | If false, do not claim pedestrian long-horizon world model. |
| Scene-Gold Gate | True | gold+silver scenes=20 | Upgrade rule-confirmed silver to human-confirmed gold where possible. |
| Per-Agent Episode Gate | True | per-agent episodes with >=2 agents=320 | Do not train multi-agent world model. Episodes are not truly multi-agent. |
| GoalBench-Gold Gate | True | official gold/silver records=1530 | If below 50, GoalBench remains diagnostic only. |
| No Leakage Gate | True | candidate goals train-only; future endpoint labels eval/train only; central velocity not used | Repair candidate-goal split policy. |
| Stage 9 Readiness Gate | True | ready | Pass data, scene, per-agent, no-leakage gates first. |

stage9_ready: `True`
latent_stage5c_ready: `False`
smc_ready: `False`
expert_audit_score: `75`
verdict: `stage8p5_ready_for_stage9_per_agent_training`
