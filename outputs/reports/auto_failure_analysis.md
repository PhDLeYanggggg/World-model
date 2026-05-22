# Auto Failure Analysis

## Top Failures

- Deterministic residual still does not beat strongest causal baseline by the required margin.
- Latent generative readiness is false; deterministic gates remain the blocker.
- SMC readiness is false; no strong stochastic proposal exists yet.
- Scene labels still rely heavily on rule-confirmed silver annotations, not human gold.

## Recommended Fixes

- Repair deterministic residual around baseline-failure cases before latent/stochastic work.
- Keep latent generative disabled and generate a plan only after deterministic gates pass.
- Keep SMC disabled until stochastic coverage improves.
- Upgrade high-value scenes from silver_rule_confirmed to silver_human_confirmed/gold_human.
