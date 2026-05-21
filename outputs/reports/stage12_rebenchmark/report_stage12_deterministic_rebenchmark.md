# Stage 12 Deterministic Re-benchmark

Datasets: `['aerialmpt', 'eth_ucy', 'eth_ucy_ewap', 'trajnet']`
Episodes >=2 agents: `660`
Verified t+10/t+50/t+100: `655/320/320`
Predicts all agents: `True`
Latent enabled: `False`
SMC enabled: `False`
Deterministic gate passed: `False`

| dataset | best variant | horizon | FDE | baseline FDE | improvement |
| --- | --- | --- | --- | --- | --- |
| aerialmpt | per_agent_no_scene | 5 | 2.746454 | 2.735528 | -0.003994 |
| eth_ucy | per_agent_scene_only | 10 | 5.157833 | 5.17126 | 0.002596 |
| eth_ucy_ewap | per_agent_no_scene | 100 | 5.460224 | 5.460224 | 0.0 |
| trajnet | per_agent_goal_only | 10 | 20.748859 | 20.771859 | 0.001107 |

## Limitations

- This is deterministic re-benchmarking, not latent generative modeling.
- Only ETH/UCY EWAP currently provides verified pedestrian long-horizon t+50/t+100.
- AerialMPT remains pixel-space because no homography or metric scale is available.
- SMC remains disabled.
