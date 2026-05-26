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
- Fresh Stage42-J static-gated full-waypoint repair showing that validation-selected partial-static experts convert the Stage42-I negative full model into positive ADE/FDE full-waypoint evidence while preserving easy cases.
- Fresh Stage42-K static-gated checkpoint training showing that a learned static gate/dropout can be trained directly into a checkpoint and improve over Stage42-I full static+sequence while preserving easy cases.
- Fresh Stage42-L horizon-aware static gate repair showing that t+50-specific gate conditioning fixes the Stage42-K ADE t50 sign while preserving easy cases.
- Fresh Stage42-M policy-distillation negative result showing that coarse domain/horizon alpha distillation is insufficient; row-level gain/harm supervision is needed.
- Fresh Stage42-N row-level gain/harm static-gate pilot showing that row-level alpha supervision improves all/hard but still fails t+50, so alpha-style gate distillation alone is insufficient.
- Fresh Stage42-O explicit gain/harm selector showing that row-level switch/gain/harm prediction improves all/hard and uses train-only normalization, but still does not pass ADE t50.
- Clear claim boundaries and no-leakage policy.

## What Is Not Yet Strong Enough

- Full retrained ablation for every named component: Stage42-G/H cover key feature/safety selector and causal sequence-history ablations, but JEPA, full Transformer, endpoint-bridge, and full-waypoint-shape retraining remain open.
- Full sequence-to-waypoint deployment: Stage42-L repairs the fresh checkpoint t50 sign, but it still underperforms the Stage42-J policy-level gate. A stronger paper claim still needs distillation of Stage42-J's domain/horizon expert selection into a fresh checkpoint, longer training, or bootstrap over the improved checkpoint.
- Policy distillation and row selector: Stage42-M shows that distilling only slice-level static alpha can improve FDE t50 but harms ADE t50. Stage42-N shows that row-level alpha/gain/harm supervision improves all/hard but still harms ADE t50. Stage42-O shows that an explicit gain/harm selector improves all/hard more cleanly under train-only normalization, but ADE t50 is still slightly negative. This branch needs t+50-specific teacher ensembles or per-domain horizon calibration before it can support a deployable checkpoint claim.
- Metric/time-calibrated pedestrian benchmark claims.
- External expansion beyond the current converted top-down state with independent legal datasets.
- Floor-free or partially floor-free neural deployment that preserves proximity/collision safety.
- Strong JEPA/full-Transformer positive contribution claim; current evidence favors protected bounded dynamics and causal sequence-history modeling over pure JEPA/Transformer.

## Shortest Next Path

1. Build a t+50-specific row-level gain/harm teacher ensemble or per-domain horizon-calibrated selector, because Stage42-O improves all/hard but still does not pass ADE t50 under strict train-only normalization.
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
