# Stage43 Locked Candidate Model Card

## Model Family

Protected multimodal latent-state candidate with safety-floor deployment.

## Inputs

- Past/current causal trajectory and density features.
- Baseline-family rollouts and protected floor predictions.
- Source/domain/horizon tokens.
- Scene/goal/interaction proxy features where legally and causally available.

## Outputs

- Protected trajectory/full-waypoint decisions.
- Failure/gain/harm/safe-switch proxy heads.
- Interaction and physical-validity diagnostic heads.

## Current Evidence

- all improvement: `50.25%`
- t50 improvement: `51.23%`
- hard/failure improvement: `47.88%`
- easy degradation: `0.00%`
- t100 raw-frame diagnostic: `0.00%`

## Deployment Boundary

The safety floor remains required. Ungated neural deployment, Stage5C execution, and SMC are not enabled.
