# Stage 11 Self-Labeled Training Report

Model type: `self_labeled_per_agent_deterministic_residual`
Datasets: `['aerialmpt', 'eth_ucy', 'trajnet']`
Episodes >=2 agents: `340`
Verified t+10/t+50/t+100: `335/0/0`
Predicts all agents: `True`
Latent enabled: `False`
SMC enabled: `False`

| dataset | best variant | horizon | FDE | baseline FDE | improvement |
| --- | --- | --- | --- | --- | --- |
| aerialmpt | per_agent_no_scene | 5 | 2.746454 | 2.735528 | -0.003994 |
| eth_ucy | per_agent_scene_only | 10 | 5.157833 | 5.17126 | 0.002596 |
| trajnet | per_agent_goal_only | 10 | 20.748859 | 20.771859 | 0.001107 |

## Limitations

- AerialMPT visual labels are AI visual silver, not human gold.
- AerialMPT is pixel-space because no homography or metric scale is available.
- Pedestrian/drone t+50/t+100 remains unavailable.
- Stage 11 is deterministic; latent generative and SMC remain disabled.
