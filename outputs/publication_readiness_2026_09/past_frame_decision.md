# Past-Only Frame Conditioning Before New Fits

## Question

The source adapter uses current ego velocity to define heading, but substitutes
angle zero when that velocity vanishes. Neural residuals can therefore learn
scene-axis conventions on static histories. Does a past-only reference frame
repair transfer without changing the approved endpoint or scientific roles?

This is a targeted conditioning control, not a new equivariant architecture or
proof that rotation sensitivity causes all observed prediction errors. Earlier
fixed-head EqMotion results remain negative and are not replaced by this test.
EqMotion explicitly addresses geometric equivariance and interaction invariance;
this is established prior work, not a novelty claim for our coordinate control.
[Author abstract](https://arxiv.org/abs/2303.10876), checked 2026-09-17. This pass
verified the abstract; direct CVF page/PDF requests returned403, so no new
full-paper reading is claimed here.

## Input-Only Frame

Use the latest nonzero observed ego velocity. If unavailable, use the nearest
neighbor with nonzero last observed displacement and valid increasing times.
If none, use the nearest supported nonzero relative neighbor position. Selection
uses the existing neighbor slot order, not labels. Direction threshold is 1e-8
in the stored feature representation. No future destinations or new scene goals.

All typed position/velocity/baseline and directed image-motion vectors rotate
to this frame. Scalar summaries, masks and observed image coverage stay unchanged.
Predicted residual vectors rotate back before adding the original CV baseline.
Without any anchor, use exact CV: no arbitrary learned direction. This fallback
does not remove queries. Image-motion vectors are not used to choose the frame.

A past-only census finds 11,601 ego-velocity anchors, 354 neighbor-velocity
anchors, 3 neighbor-position anchors and 8 unsupported rows. All365 static
histories remain. Counts do not use targets or determine thresholds.

## Fixed Comparison

- Original features/head: 18 exact-replayed frozen row/log controls.
- Guard-only: 18 new fits with unchanged axes and the same no-anchor CV rule.
- Past-frame: 18 new fits with frame-conditioned features and the same CV rule.

Guard-only separates coordinate conditioning from the eight-row fallback.
Both quality-control and directed-motion variants, seeds17/29/43, the original
three physical-scene fit folds, row-uniform batches, 4,000 updates, log1p ADE,
AdamW settings and model parameter count are fixed. Final checkpoint only;
no model/epoch/threshold selection on held outcomes. Partial pilot resumes with
optimizer/RNG state and no held evaluation before completion.

Training normalization uses only that fold's training features. The loss and
evaluation operate on original coordinates in the frozen past-normalized frame.
Primary ADE, equal physical-scene aggregation and easy limit2% stay unchanged.
Report all arms/seeds/scenes, native diagnostics separately by recording,
anchor support, static-start/stay slices, losses and compute. Three-scene
resampling is descriptive, not independent confirmation.

Additionally re-express all held feature vectors and baseline paths by quarter
turns90/180/270 degrees, holding training moments fixed. Compare restored
predictions with originals without reading targets. This checks downstream
coordinate consistency of typed summaries, not image-rotation invariance of
the optical-flow/cropping pipeline or useful forecasting accuracy.

## Boundaries

All11,966 previously exposed fit queries stay. Students, development, calibration
and confirmation roles remain closed. No new source admission, latent generation,
SMC, sensor-as-of claim, metric/time claim or deployment. Offline annotations
can contain retrospective interpolation. A consistent but inaccurate model is
still a negative forecasting result. More equivariance is not itself novelty.
