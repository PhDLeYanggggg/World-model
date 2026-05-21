# Stage 6 Model Card

Model family: deterministic baseline-failure-aware trajectory world-state model.

Not enabled: latent generative modeling, diffusion, CVAE, SMC.

Components:

- Baseline failure predictor from causal past-window features.
- Failure-aware gated residual model: `baseline + alpha * bounded_residual`.
- Interaction ablations: no interaction, scalar interaction, graph interaction.

Failure predictor test AUROC: `0.899098`.
Failure predictor test AUPRC: `0.694048`.

Graph interaction did not pass the interaction gate. Scalar/graph features are kept for diagnostics, not claimed as a solved interaction model.

Gate result: `5 / 10`.
Verdict: `stage6_failure_bench_built_but_not_stage5c_ready`.
