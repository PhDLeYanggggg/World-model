# Stage 9 Model Card

Model type: deterministic per-agent multi-agent bounded residual over strongest causal baseline.
Prediction form: `prediction_i = baseline_i + alpha_i * bounded_residual_i`.
Predicts all active agents with masks. No latent generative branch. No SMC.

Gate verdict: stage9_per_agent_training_done_not_stage5c_ready
Expert audit score: 75
