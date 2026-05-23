# Stage 25 Selector Model Card

- Model family: regret-minimizing, confidence-gated baseline policy.
- Inference features: causal metadata only from Stage24 eval tables; no future endpoint, no oracle residual, no central velocity.
- Deployment: only if gates pass; otherwise BPSG-MA v1 strongest-baseline fallback remains deployable.
- Coordinate/horizon: SDD pixel-space raw-frame.
