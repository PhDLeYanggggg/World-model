# Stage28 Model Card

- Model name: M3W-LAS, latent-augmented selector over physical causal baselines.
- Inputs: Stage26 causal features plus frozen M3W JEPA/Transformer/Hybrid hidden features.
- Outputs: selected physical baseline and confidence/fallback diagnostics.
- Deployment: Stage26 remains deployable unless Stage28 candidate_v2 is true.
- Not true 3D, not foundation-scale, not latent generative, no SMC.
