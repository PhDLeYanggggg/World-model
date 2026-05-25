# Stage42 A-Journal Gap Analysis

## Current Position

Stage42 is strong enough to support a serious protected 2.5D external world-state dynamics manuscript draft. It is not yet enough for a broad true-3D/foundation/world-model claim.

## What Is Already Paper-Usable

- Fresh source-level external validation.
- Fresh full-waypoint all-agent world-state evaluation.
- Fresh safety-floor study showing why ungated neural cannot be deployed.
- Fresh Stage42-G Phase1 retrained external selector ablations for history, neighbor/interaction, goal/scene, domain expert, safe-switch, and teacher-floor proxy variants.
- Fresh Stage42-H causal temporal sequence ablation showing that history tokens are strongly positive when encoded as a sequence rather than flattened into a ridge-selector feature vector.
- Fresh Stage42-I sequence-to-full-waypoint experiment showing that causal history gives a small positive full-waypoint contribution, while un-gated static/context features currently hurt protected ADE.
- Clear claim boundaries and no-leakage policy.

## What Is Not Yet Strong Enough

- Full retrained ablation for every named component: Stage42-G/H cover key feature/safety selector and causal sequence-history ablations, but JEPA, full Transformer, endpoint-bridge, and full-waypoint-shape retraining remain open.
- Full sequence-to-waypoint deployment: Stage42-I is partial because the full static+sequence model is ADE-negative; the no-static sequence variant is positive, so static/context gating must be repaired before making a stronger full-waypoint claim.
- Metric/time-calibrated pedestrian benchmark claims.
- External expansion beyond the current converted top-down state with independent legal datasets.
- Floor-free or partially floor-free neural deployment that preserves proximity/collision safety.
- Strong JEPA/full-Transformer positive contribution claim; current evidence favors protected bounded dynamics and causal sequence-history modeling over pure JEPA/Transformer.

## Shortest Next Path

1. Repair Stage42-I full-waypoint dynamics with static-gated/static-dropout causal sequence heads, because no-static-context is positive while the unconditional static+sequence model is ADE-negative.
2. Run Stage42-G/H Phase2 true retrained ablations for no-JEPA, no-Transformer, no-endpoint-bridge, and no-full-waypoint-shape with bootstrap or three seeds; Stage42-H has repaired the history-token question with an actual sequence model, so the next ablation priority is full Transformer/JEPA/full-waypoint-shape rather than flattened-history.
3. Add one more legally verified external top-down pedestrian/drone dataset or a stronger held-out source split.
4. Build a proximity-safe internal self-gate that reduces teacher-floor dependence without increasing collision/proximity risk.
5. Obtain verified homography/FPS/stride for at least one pedestrian subset, or keep all claims raw-frame/dataset-local.

## Absolute Non-Claims

- Not true 3D.
- Not foundation.
- Not metric/seconds-level pedestrian prediction.
- Not Stage5C or SMC.
- Not ungated neural deployment.
