# Stage 12 Failure Analysis

Stage 12 fixed the most important data-side blocker by adding `eth_ucy_ewap` as a verified pedestrian long-horizon source. It also expands the benchmark to 660 multi-agent episodes, 320 verified t+50/t+100 episodes, 43 scene packs, and 5574 GoalBench v4 records.

The remaining failure is model-side:

1. The deterministic residual model still does not beat strongest causal baselines by 5%.
2. On `eth_ucy_ewap` t+100, the best learned model matches but does not improve over the baseline.
3. Scene/goal features are still mostly weak silver/rule labels, not high-quality human gold.
4. AerialMPT is useful for visual scene annotation, but remains pixel-space and short-horizon without homography or scale.
5. Stage 5C latent generative and SMC remain premature because the deterministic proposal is not strong enough.

Recommended fixes before latent modeling:

1. Train a stronger Stage 13 deterministic model specifically on EWAP t+50/t+100 with failure-aware gating.
2. Add more true scene-goal annotations and use ETH/UCY destinations/H maps more explicitly.
3. Add Stanford Drone Dataset or OpenTraj local data with real scene images and longer pedestrian/drone tracks.
