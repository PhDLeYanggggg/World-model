# Stage 8 Model Card

Model type: deterministic scene/goal-conditioned bounded residual over strongest causal baseline.

Not true 3D. Not latent generative. Not SMC. Top-k goal diagnostics are deterministic candidate evaluations only.

Gate verdict: stage8_scene_goal_multiagent_scaffold_not_stage5c_ready
Expert audit score: 71

Official prediction form:

`prediction = strongest_causal_baseline + alpha * bounded_residual(goal, scene, multi_agent_context)`

Known limits: inferred goals, no verified pedestrian/drone t+50/t+100 in this run, primary-agent residual target, weak interaction trajectory evidence.
