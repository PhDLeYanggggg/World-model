# Auto Gate Report

| gate | pass | evidence |
| --- | --- | --- |
| Data Gate | True | At least one real pedestrian/drone dataset loaded and converted. |
| Long-Horizon Gate | True | At least one pedestrian/drone source supports verified t+50/t+100. |
| Annotation Gate | True | At least 3 human-confirmed or high-quality silver scenes exist. |
| Scene/Goal Gate | False | GoalBench beats majority or improves calibrated goal metrics. |
| Multi-Agent Gate | True | At least 300 multi-agent episodes with >=2 agents. |
| Strong Baseline Gate | True | Strongest causal baseline computed. |
| Deterministic Improvement Gate | False | Learned deterministic model beats strongest causal baseline. |
| Easy Preservation Gate | False | Easy subset degradation <=2% is not proven by the latest report. |
| Scene/Goal Ablation Gate | False | Scene/goal ablation is not proven to improve trajectory metrics. |
| Interaction Gate | False | Interaction module does not yet improve hard/failure trajectory metrics. |
| Physical Validity Gate | True | No major physical-validity degradation reported in Stage 12 summary. |
| Latent Generative Readiness Gate | False | Must pass deterministic scene/goal/interaction gates first. |
| SMC Readiness Gate | False | No stochastic proposal with coverage lift exists. |

latent_generative_ready: `False`
smc_ready: `False`
