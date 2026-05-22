# Auto Deterministic Training Report

- executed_training: `False`
- prediction_form: `prediction_i,h = strongest_causal_baseline_i,h + alpha_i,h * bounded_residual_i,h`

## Training Order

- strongest causal baselines
- alpha-only baseline failure gate
- bounded residual without scene/goal
- scene-only
- goal-only
- interaction-only
- scene+goal
- scene+interaction
- goal+interaction
- full scene+goal+interaction
- hard/failure fine-tuned
- long-horizon fine-tuned
