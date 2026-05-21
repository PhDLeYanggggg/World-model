# Stage 6 Gates

Passed: 5 / 10

| gate | pass | evidence | next fix |
| --- | --- | --- | --- |
| Pedestrian/Drone Long-Horizon Gate | False | 0 actual pedestrian/drone sources support verified t+50/t+100 | No pedestrian long-horizon claim without this. |
| HardBench Reliability Gate | True | HardBench-v1 hard episodes=53, eligibility=official | Need at least 50 official hard episodes. |
| BaselineFailureBench Gate | True | failure_samples=48, train_ok=True, eval_ok=True | Need enough failure samples for train/test. |
| Failure Predictor Gate | True | AUROC=0.899098, AUPRC=0.694048, positive_rate=0.302632 | Improve causal failure predictor. |
| Alpha Calibration Gate | True | easy=0.004620422647958356, hard=0.03775980526559732, failure=0.04468315373365586, corr=0.07206179273714433 | Alpha should increase from easy to hard to failure. |
| Failure-Aware Improvement Gate | False | best failure subset improvements={'eth_ucy': 0.098915, 'tgsim': -0.0048, 'tgsim_i90': 0.013633, 'trajnet': 0.0} | Need >=10% improvement on BaselineFailureBench. |
| Easy Preservation Gate | True | best easy improvements={'eth_ucy': 0.0, 'trajnet': 0.0} | Do not degrade easy cases. |
| Verified Long-Horizon Gate | False | verified long-horizon improvements={'tgsim': -0.010211, 'tgsim_i90': 0.011623} | Need >=5% improvement on verified t+50/t+100. |
| Interaction Gate | False | no_interaction=0.109813, interaction=0.117277 | Interaction features must help hard/failure subsets. |
| Stage 5C Readiness Gate | False | Do not enter Stage 5C. Deterministic failure-aware gates did not pass. | Pass Stage 6 deterministic gates first. |

latent_stage5c_ready: `False`
smc_ready: `False`
expert_audit_score: `70`
verdict: `stage6_failure_bench_built_but_not_stage5c_ready`
