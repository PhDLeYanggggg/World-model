# Model Card: BPSG-MA World Model v1

- true_3D: false
- foundation_world_model: false
- latent_generative: false
- SMC: false
- prediction_form: prediction = strongest_baseline + alpha * bounded_residual
- deployment: strongest baseline fallback with diagnostics
- official_horizon: t+50
- t+100: diagnostic only
