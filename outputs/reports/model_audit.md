# Stage 2 Model Audit

This audit describes the current `pseudo3d_world_model.py` system before the learned stage-2 upgrade.

## Summary

The current system is **not** a full learned world model. It is a camera-aware 2.5D **state-space world-model scaffold** with hand-written transition rules, weak calibration, SMC rollout, collision projection, and diagnostic reporting.

Important limitations:

- AerialMPT `bauma3` has only 16 frames in the selected sequence.
- From start frame 4, the maximum observed evaluation horizon is `t+12`.
- Current `t+100` has **no ground truth** on AerialMPT and can only be treated as free-run trend exploration.
- Current terminal clustering has weak semantic diversity: the three terminal clusters are all `jammed / east-shifted`.

## 1. Real Observed Data

These values come directly from data files or image metadata:

- AerialMPT image frames.
- MOT-format pedestrian annotations: `frame_ID`, `tracking_ID`, `x1`, `y1`, `w`, `h`.
- Track identity within each sequence.
- Frame index and approximate FPS used by the experiment.
- Image width and image height.

These are the only directly observed variables in the real-data stage.

## 2. Derived From Observation

These variables are computed from observed image annotations:

- Image point measurement `u, v`, currently approximated from bbox center.
- Weak ground-plane coordinates `X, Y` via homography.
- Approximate velocity `Vx, Vy` from adjacent frame differences.
- Approximate acceleration from finite differences or transition integration.
- Pixel-space / pseudo-metric centroid.
- Local density under the assumed scale.
- Min-gap estimates after converting body radius to meters.
- Projection error under the weak homography.

These are not independently observed physical quantities. Their correctness depends on tracking quality, frame timing, and calibration assumptions.

## 3. Latent Inferred Variables

These variables are not directly measured and are inferred or sampled:

- Per-person body radius.
- Per-person body height.
- Per-person mass.
- Desired speed.
- Desired direction.
- Latent goal point or goal region.
- Group id.
- State uncertainty covariance.
- Future terminal event mode.

These variables can be useful for modeling, but they should not be presented as real measured ground truth.

## 4. Manual Assumptions

The current model relies on these assumptions:

- Ground is flat: `Z = ground_height(X,Y) = 0`.
- No real camera intrinsics/extrinsics are available.
- No homography control points are available.
- Weak homography is built from a rough GSD / meter-per-pixel scale.
- AerialMPT `bauma3` rough scale is set to `0.105 m/px`, with range `[0.08, 0.13]`.
- Body radius defaults to about `0.30 m`.
- Body height defaults to about `1.70 m`.
- Scene obstacles are manually approximated as polygons.
- Exit / goal regions are manually approximated.
- Observation noise uses a fixed pixel sigma.

These are engineering priors, not learned facts.

## 5. Hand-Written Physics Rules

The current transition model is mostly hand-written:

- Inertia / velocity integration.
- Goal-seeking desired velocity.
- Social repulsion between nearby people.
- Group cohesion.
- Obstacle repulsion.
- Boundary repulsion.
- Speed clamping.
- World-space collision projection.
- Scene constraint projection.
- Projection cost and scene violation cost as particle weight penalties.

These rules create physically plausible rollouts, but they are not learned from data.

## 6. Truly Learned From Data

In `pseudo3d_world_model.py`, very little is learned:

- Terminal clustering uses KMeans over particle terminal features, but this is unsupervised post-processing rather than a learned transition model.
- Earlier files contain a Social-MLP / Transformer pixel-transition model, but the current pseudo-3D scaffold does not use a learned world-coordinate transition head.

Therefore the current pseudo-3D system is **not yet a learned world model**.

## 7. Current Evaluation Status

Synthetic data:

- Not yet present before stage 2.
- No long-horizon ground truth existed for validating `t+100`.

AerialMPT:

- `bauma3` can be evaluated only up to `t+12` from the chosen start.
- `t+100` must be labeled as free-run only.
- Any `t+100 ADE/FDE` claim on this sequence would be invalid.

## 8. Known Failure Modes

- Terminal clusters are semantically under-diverse: all three current clusters are `jammed / east-shifted`.
- Weak calibration means metric quantities have substantial scale uncertainty.
- Manual obstacle polygons are too rough for strong scene-understanding claims.
- Latent goals are inferred from short history and may collapse to local motion.
- Collision projection can enforce non-overlap even when raw point annotations look crowded.
- Without long real trajectories, long-horizon trend quality cannot be proven on AerialMPT.

## Stage-2 Requirement

The next step should introduce a controlled SyntheticPhysicalCrowd2.5D dataset with real `t+100` ground truth, then train a neural residual world transition model and compare:

- hand-coded physics proposal,
- learned neural proposal,
- physics + neural residual proposal.

Only after that comparison should the model be described as a learned 2.5D world model.
