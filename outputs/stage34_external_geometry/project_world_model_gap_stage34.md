# Stage34 Project World Model Gap

- Stage34 adds per-row external geometry, train-only goals, relative baselines v2, geometry-aware selectors, and external transfer diagnostics.
- Positive cross-domain transfer requires Gate5 and Gate11, not merely fallback 0.0.
- If gates fail, the blocker is now narrowed to external row/scene/horizon/data-length limits rather than missing row geometry.
