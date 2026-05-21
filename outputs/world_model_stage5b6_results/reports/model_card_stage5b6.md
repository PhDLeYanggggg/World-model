# Stage 5B.6 Model Card

Model type: baseline-aware deterministic gated residual over each dataset's strongest causal baseline.

Prediction form: `prediction = strongest_causal_baseline + alpha * bounded_residual`.

The model is not latent generative, not SMC, and not a true 3D world model. It remains a 2.5D / trajectory world-state benchmark model.

Implemented variants:

- `gated_residual_all_data`
- `gated_residual_hard_weighted`
- `gated_residual_failure_classifier_aux`
- interaction ablations: no interaction, nearest-neighbor scalar, graph interaction, graph temporal history

Alpha calibration: corr=0.207346, easy_alpha=0.029783, hard_alpha=0.091664.

Interaction result: graph interaction did not beat no-interaction in the quick hard benchmark. This is a failure, not a success.

Gate result: 3 / 10, verdict `stage5b6_reliability_repaired_but_deterministic_gate_failed`.
