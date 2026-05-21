# Stage 5B Deterministic Model Card

Model: `deterministic_linear_residual_over_strongest_causal_baseline`.

This is a gated deterministic pretraining test, not a latent generative model, not SMC, and not a large-scale foundation model. It learns a small linear residual over each dataset's strongest causal baseline.

Official inputs: causal finite-difference position-derived velocity and past states only.

Known limitation: the residual model is dataset-specific in this quick run; true leave-one-dataset-out transfer is diagnostic only.
