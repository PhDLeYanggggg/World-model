# SAM-JEPA-2.5D Model Spec

- context_encoder: trajectory MLP/TCN-style features + scene raster summary + interaction density proxy
- target_encoder: future trajectory/interaction latent encoder with stop-gradient semantics in this lightweight implementation
- predictor: MLP/ridge predictor from context latent to target latent
- multimodal_fusion: concatenation + normalized projection; no GPT-style autoregressive Transformer
- losses: latent L2, cosine proxy, variance/covariance non-collapse checks, temporal consistency proxy, cross-modal alignment proxy
- forbidden: pixel reconstruction, diffusion, latent generative rollout, SMC
