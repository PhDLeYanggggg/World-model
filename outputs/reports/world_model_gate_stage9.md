# Stage 9 Gates

Passed: 3 / 11

| gate | pass | evidence | next fix |
| --- | --- | --- | --- |
| Per-Agent Data Gate | True | >=2 agent episodes=320 | Need at least 300 official per-agent multi-agent episodes. |
| No Leakage Gate | True | leakage_flags={'test_endpoints_used_for_goals': False, 'future_endpoint_used_as_input': False, 'central_velocity_used': False, 'scene_split_leakage_detected': False} | Candidate goals must be train-only; no future endpoint inputs. |
| Strong Baseline Gate | True | datasets=['eth_ucy', 'trajnet'] | Every official dataset/scene needs strongest causal baseline. |
| Per-Agent Model Gate | False | full all-test mean improvement=-0.001592 | Do not enter Stage 5C. Per-agent deterministic world model is not strong enough. |
| Hard/Failure Gate | False | full hard/failure best improvement=0.000537 | Need >=10% on hard or baseline-failure subset. |
| Easy Preservation Gate | False | full easy mean improvement=-999.0 | Do not degrade easy subset. |
| Interaction Gate | False | full minus scene_goal hard/failure gain=-0.000414 | Interaction must improve trajectory metrics, not just auxiliary. |
| Scene/Goal Gate | False | scene_goal minus no_scene gain=-5e-06 | Scene/goal must improve hard/failure or GoalBench subset. |
| Multi-Agent Gate | False | full minus no_scene on >=5 agents=-0.003381 | Per-agent all-agent model must beat primary/simple fallback on multi-agent scenes. |
| Verified Long-Horizon Gate | False | t50=0 t100=0 | Do not claim pedestrian long-horizon world model. Need verified t+50/t+100 pedestrian/drone data. |
| Stage 5C Readiness Gate | False | Do not enter Stage 5C. Per-agent deterministic gates and verified long-horizon gate are not satisfied. | Pass deterministic and long-horizon gates before latent generative work. |

latent_stage5c_ready: `False`
smc_ready: `False`
expert_audit_score: `75`
verdict: `stage9_per_agent_training_done_not_stage5c_ready`
