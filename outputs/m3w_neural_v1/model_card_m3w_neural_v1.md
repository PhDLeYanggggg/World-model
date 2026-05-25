# M3W-Neural v1 Model Card

## Intended Use

Protected 2.5D multi-agent trajectory world-state diagnostics and external top-down selector/dynamics research under a Stage37 safety floor.

## Not Intended For

- Metric 3D prediction.
- Seconds-level physical claims.
- Autonomous deployment without external safety review.
- Stage5C latent generative rollout.
- SMC inference.

## Model Family

Composite-tail safe-switch bounded neural dynamics with causal past-only features, gain/harm/tail-risk gating, and fallback to the Stage37/teacher safety floor.

## Safety Floor

If confidence/gain/harm/tail-risk safety does not permit a switch or low-risk bounded blend, the model falls back to Stage37/source-rotation baseline behavior.
