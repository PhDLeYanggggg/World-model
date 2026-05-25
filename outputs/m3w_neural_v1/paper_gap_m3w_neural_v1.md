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
- Pure UCY-only retrain/select/test evidence.
- Full all-agent continuous world-state rollout beyond protected endpoint interpolation.

## Shortest Next Path

1. Run a stricter pure UCY-only retrain/select/test protocol if another independent UCY-like source becomes available.
2. Extend from bounded endpoint/tail interpolation to full multi-step all-agent world-state rollouts.
3. Complete homography/FPS/scale audit before any physical-world claims.
