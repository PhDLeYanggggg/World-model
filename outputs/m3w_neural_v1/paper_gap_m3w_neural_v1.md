# M3W-Neural v1 Paper Gap

## Evidence That Can Be Claimed

- A protected composite-tail bounded neural dynamics candidate beats the Stage37/source-rotation safety floor on external all/t+50/t+100/hard-failure metrics with easy preservation.
- It has positive bootstrap CI lows, three seed-aware replications, and pure-UCY source-heldout support.
- Endpoint/FDE geometry alignment is audited.
- Stage5C and SMC remain disabled.

## Evidence That Cannot Be Claimed Yet

- True 3D or metric world modeling.
- Foundation-scale world model.
- Seconds-level long-horizon prediction.
- Ungated neural dynamics safe replacement.
- Independent UCY-like replication beyond the current strict pure-UCY zara held-out sources.
- Ungated learned waypoint-shape dynamics: calibrated learned-shape residuals are positive on two domains, but the contribution is small and protected by endpoint bridge/floor fallback.
- Ungated full-row all-agent continuous world-state rollout without the Stage37/teacher safety floor.
- Residual source-switching over the fixed composer as a deployable improvement path.

## Shortest Next Path

1. Add independent UCY-like validation sources to validate the repaired strict pure UCY-only neural retrain beyond the current zara held-out sources.
2. Strengthen the protected all-agent full-waypoint rollout with stricter source-heldout retrain/select/test evidence and safer no-fallback neural rollout research.
3. Complete homography/FPS/scale audit before any physical-world claims.
4. Add genuinely new scene/domain context before retrying fixed-composer residual source-switching.
