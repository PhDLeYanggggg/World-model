# Stage 5B Next Steps

1. Add a real pedestrian/drone source with verified t+100, preferably full SDD or a full ETH/UCY/OpenTraj conversion with longer tracks and scene homographies.
2. Replace the quick linear residual with a real deterministic temporal-interaction model, but keep residual-over-strongest-baseline as the training target.
3. Add real scene geometry or lane/map constraints for datasets that support it; do not report off-road or obstacle metrics where maps are absent.
4. Run true leave-one-dataset-out training only after the deterministic model beats strongest causal baselines in-domain.
5. Keep latent generative and SMC disabled until Stage 5B Gates 1-7 pass.
