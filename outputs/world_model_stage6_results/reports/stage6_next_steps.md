# Stage 6 Next Steps

1. Add an actual legal pedestrian/drone long-horizon source, preferably SDD or full OpenTraj/ETH-UCY with verified t+50/t+100.
2. Convert multi-agent episodes instead of one-primary-agent windows so interaction encoders model real neighboring trajectories.
3. Train the failure-aware residual only on reliable BaselineFailureBench folds and require >=10% failure-subset improvement before any Stage 5C latent generative work.
