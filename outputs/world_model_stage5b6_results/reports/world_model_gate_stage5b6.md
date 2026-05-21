# Stage 5B.6 Gates

Passed: 3 / 10

| gate | pass | evidence | next fix |
| --- | --- | --- | --- |
| Hard Reliability Gate | False | 0 hard subsets are official gate eligible (>=50 hard episodes) | Collect more hard episodes; do not gate on one-episode wins. |
| Pedestrian / Drone Horizon Gate | False | 0 pedestrian/drone sources have verified t+50/t+100 | Prepare SDD/full OpenTraj/long AerialMPT with legal access. |
| No Leakage Gate | True | 4/4 leakage audits passed | Keep official features causal. |
| Gated Residual Gate | False | gated residual beats strongest baseline by >=5% on 1 dataset target horizons | Improve baseline failure detection and residual training. |
| Hard Interaction Gate | False | gated residual beats strongest baseline by >=10% on 0 official hard subsets | Need reliable hard subsets and stronger interaction model. |
| Alpha Calibration Gate | True | corr=0.207346, easy_alpha=0.029783, hard_alpha=0.091664 | Alpha should rise when baseline likely fails and stay low on easy segments. |
| Interaction Encoder Gate | False | hard improvement no_interaction=0.04006, graph=-0.013418 | Graph interaction must beat no-interaction on hard subsets. |
| Verified Long-Horizon Gate | False | 0 verified t+100 sources beat strongest baseline by >=5% | Need at least one robust long-horizon win. |
| Physical Validity Gate | True | minimum physical validity remains >=0.95 | Bound residual and add validity losses. |
| Stage 5C Readiness Gate | False | Do not enter Stage 5C. Deterministic gated interaction model is not strong enough. | Pass deterministic reliability gates first. |

latent_stage5c_ready: `False`
smc_ready: `False`
expert_audit_score: `68`
verdict: `stage5b6_reliability_repaired_but_deterministic_gate_failed`
