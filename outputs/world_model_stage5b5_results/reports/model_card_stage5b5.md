# Stage 5B.5 Model Card

Models evaluated:

1. `numpy_temporal_interaction_ridge_residual` with causal history features, nearest-neighbor interaction features, domain flags, horizon conditioning, residual clipping, and validation-selected residual gate alpha.
2. PyTorch deterministic temporal-interaction variants: `direct_multi_horizon`, `recurrent_rollout`, and `hybrid`.

The PyTorch GRU temporal-interaction path now runs in the cleaned `.venv_m3_torch` environment and produced three checkpoints: direct multi-horizon, recurrent rollout, and hybrid.

The PyTorch backend recovery is an engineering fix, not a model-success result. The deterministic learned model still does not beat the strongest causal baseline on enough all-test / hard-test / verified t+100 benchmarks.

This stage remains deterministic. It is not CVAE, diffusion, latent generative modeling, or SMC. It predicts residuals over each dataset's strongest causal baseline, never over weak hand physics.
