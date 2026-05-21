# Stage 5B.5 Next Steps

1. Add a true long-horizon pedestrian/drone source: SDD with accepted license, full OpenTraj/ETH-UCY if legally prepared, or AerialMPT longer sequences with verified trajectories.
2. Scale and repair the PyTorch GRU/Transformer temporal-interaction model now that the runtime can complete, especially multi-agent scene batching, stronger hard-subset training, and stable long-horizon residual gating.
3. Build multi-agent episodes with split-safe agent groups so interaction features are model inputs, not only hard-subset diagnostics.
