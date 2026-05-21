# Stage 7 Model Card

Model family: deterministic scene/goal-grounded baseline-aware world-state predictor.

Not enabled: latent generative modeling, CVAE/diffusion, SMC.

Prediction form: `strongest_causal_baseline + alpha * goal_conditioned_bounded_residual`.

Goal predictor test top3: `0.782609` vs majority `0.826087`.
Best Stage 7 failure predictor AUROC: `0.943396`.
Interaction auxiliary trajectory lift claimed: `False`.
Gate result: `5 / 10`.
Verdict: `stage7_scene_goal_grounding_built_but_not_stage5c_ready`.
