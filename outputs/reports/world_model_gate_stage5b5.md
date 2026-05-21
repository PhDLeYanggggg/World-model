# Stage 5B.5 Gates

Passed: 6 / 10

| gate | pass | evidence | next fix |
| --- | --- | --- | --- |
| Pedestrian / Drone Data Gate | False | 0 pedestrian/drone sources support raw t+50/t+100; audit documents why current TrajNet/ETH fallback cannot | Prepare SDD/full OpenTraj/real long pedestrian tracks. |
| Hard Subset Gate | True | 3 datasets have enough hard examples for evaluation | Mine more hard examples or increase real data. |
| No Leakage Gate | True | 4/4 leakage audits passed | Fix split/causal feature issues. |
| Deterministic Model Gate | False | temporal model beats strongest baseline by >=5% on 1 dataset target horizons | Improve deterministic model before Stage 5C. |
| Hard Interaction Gate | False | temporal model beats strongest baseline by >=10% on 1 hard subsets | Improve interaction modeling and hard training. |
| Long Horizon Gate | True | temporal model beats strongest baseline by >=5% on 1 verified t+100 sources | Need at least one robust verified t+100 win. |
| Physical Validity Gate | True | No collision/speed/acceleration degradation measured in this single-agent quick benchmark | Add multi-agent physical validity once multi-agent episodes return. |
| Stability Gate | True | max residual magnitude=0.631496, max gate alpha=0.45 | Keep residual clipping/gating; reject exploding rollout. |
| Cross Dataset Gate | True | diagnostic cross-dataset report exists | Run leave-one-dataset-out once model is real. |
| Stage 5C Readiness Gate | False | Do not enter Stage 5C latent generative. Deterministic interaction model is not strong enough. | Pass Gates 1-8 first. |

latent_stage5c_ready: `False`
smc_ready: `False`
expert_audit_score: `70`
verdict: `stage5b5_hard_benchmark_built_but_deterministic_gate_failed`
