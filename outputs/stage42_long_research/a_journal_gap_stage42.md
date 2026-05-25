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
- Clear claim boundaries and no-leakage policy.

## What Is Not Yet Strong Enough

- Full retrained ablation for every named component: Stage42-G/H cover key feature/safety selector and causal sequence-history ablations, but JEPA, full Transformer, endpoint-bridge, and full-waypoint-shape retraining remain open.
- Full sequence-to-waypoint deployment: Stage42-J repairs the static/context gate at policy level, but it uses cached Stage42-I checkpoints with fresh gating/eval. A stronger paper claim still needs fresh static-gated checkpoint training or bootstrap over the gated policy.
- Metric/time-calibrated pedestrian benchmark claims.
- External expansion beyond the current converted top-down state with independent legal datasets.
- Floor-free or partially floor-free neural deployment that preserves proximity/collision safety.
- Strong JEPA/full-Transformer positive contribution claim; current evidence favors protected bounded dynamics and causal sequence-history modeling over pure JEPA/Transformer.

## Shortest Next Path

1. Train a fresh static-gated/static-dropout checkpoint version of Stage42-J, because Stage42-J proves the gate works but currently uses cached Stage42-I checkpoints.
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
