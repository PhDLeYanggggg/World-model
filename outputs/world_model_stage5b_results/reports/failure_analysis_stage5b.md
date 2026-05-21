# Stage 5B Failure Analysis

The deterministic learned residual did not pass the learned dynamics gate. It beat the strongest causal baseline by at least 5% only on the short-horizon ETH/UCY fallback subset, not on two actual verified t+100 sources.

Main failure modes:

1. Traffic trajectories are very smooth under causal constant velocity, so a small residual head easily over-corrects.
2. The converted pedestrian sources in this quick run are short TrajNet-format snippets and cannot verify t+100.
3. No real scene maps, lane graphs, goals, or interaction labels were available in the official converted quick benchmark.
4. The residual model is too small and dataset-specific to be a foundation world model.

Failed or insufficient datasets:

| dataset | domain | actual_verified_t100 | official_horizons | target_horizon | strongest_causal_baseline | baseline_FDE_target | best_learned | learned_FDE_target | learned_improvement | learned_beats_5pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tgsim | traffic | True | [1, 10, 25, 50, 100] | 100 | constant_velocity_causal_fd | 6.062032 | deterministic_residual_one_step | 6.060675 | 0.000224 | False |
| tgsim_i90 | traffic | True | [1, 10, 25, 50, 100] | 100 | constant_velocity_causal_fd | 10.327657 | deterministic_residual_one_step | 10.329037 | -0.000134 | False |
| trajnet | pedestrian | False | [1, 10] | 10 | constant_velocity_causal_fd | 1.434586 | deterministic_residual_one_step | 1.439945 | -0.003736 | False |

