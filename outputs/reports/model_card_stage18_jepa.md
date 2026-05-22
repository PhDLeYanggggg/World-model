# Model Card: SAM-JEPA-2.5D

- model role: multimodal representation pretraining
- true_3D: false
- foundation_world_model: false
- latent_generative_rollout: false
- SMC: false
- architecture: trajectory/raster/context encoders + latent predictor; no next-token Transformer and no pixel reconstruction
- downstream allowed: selector, failure predictor, goal predictor, correction specialist, physical validity diagnostics
