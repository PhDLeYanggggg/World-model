# Model Audit Stage 2

## One-Sentence Status

当前系统是 pseudo-3D physics-informed SMC state-space world-model scaffold。
它不是 true 3D world model。
它不是 learned neural world model in the full real-data sense。
它的 t+100 在 AerialMPT bauma3 上只是 free-run，不是 verified forecast。

## 1. Real Observed Variables

- AerialMPT image-space detections / track positions: `u`, `v` when available.
- Short observed trajectory window in bauma3: selected slice has 16 frames.
- SyntheticPhysicalCrowd2.5D true states: generated `X`, `Y`, velocity, acceleration, goal, radius, active/reached flags, collision/obstacle/boundary diagnostics.
- Synthetic scene geometry: walkable bounds, obstacle rectangles, exits, spawn regions.

## 2. Derived From Observation

- Weak ground-plane `X/Y` from image `u/v` using weak GSD / homography assumptions in the previous pseudo3D stage.
- Velocity and acceleration estimated by finite differences.
- Heading from velocity direction.
- Local density, nearest-neighbor distance, min gap, obstacle distance, boundary distance.
- Pixel/world projection error in the previous AerialMPT scaffold.

## 3. Latent Inferred Variables

- Per-person intent / goal when real goals are not annotated.
- Desired speed and interaction strength for real AerialMPT.
- Body footprint uncertainty when no calibrated body scale is available.
- SMC particle weight, latent branch identity, and terminal semantic mode.

## 4. Human Assumptions

- Ground is modeled as `Z=0`; this is 2.5D / pseudo-3D, not recovered metric 3D.
- Pedestrians are vertical cylinders/capsules with approximate radius.
- Weak homography / GSD replaces real intrinsics/extrinsics unless camera calibration exists.
- Synthetic scenes are simplified rectangles, exits, walls, and obstacles.
- Quick demo uses fewer episodes and particles than the full config to run on CPU.

## 5. Hand-Written Physics Rules

- Social-force-like acceleration toward goals.
- Neighbor repulsion and comfort margin.
- Obstacle repulsion and boundary pushback.
- Collision projection in world coordinates.
- Obstacle/boundary projection.
- Speed and acceleration clipping.

## 6. Learned From Data

- Stage 2 trains deterministic and stochastic neural residual transition models on synthetic trajectories.
- The learned component predicts residual acceleration:

```text
A_residual = A_true_next - A_hand_physics
```

- It is not a fully learned dynamics model because hand-coded physics, constraints, and projection remain central.

## 7. Why This Is Not a Full Learned World Model

- Real AerialMPT data does not provide long t+100 supervision in the selected bauma3 slice.
- Camera calibration is weak; no real metric 3D is recovered.
- Dynamics still depend on hand-coded social force and projection.
- Scene geometry is manually/synthetically specified, not inferred from pixels.
- Goal and intent are latent in real data rather than directly observed.

## 8. Why AerialMPT bauma3 t+100 Cannot Be Evaluated

- The selected sequence has only 16 frames.
- From start frame 4, the maximum ground-truth future is t+12.
- t+20, t+50, and t+100 have no ground truth in this slice.
- Therefore any AerialMPT t+100 output is qualitative free-run only and must not be reported as ADE@100/FDE@100.

## 9. Why Previous Terminal Clusters Lacked Semantic Diversity

- The previous clusters were mostly endpoint and local event based.
- SMC branches were dominated by the same jammed / east-shifted mode.
- Without long-horizon ground truth, goal labels, and diverse scene events, terminal modes collapsed into similar congestion-risk narratives.
- Stage 2 uses semantic event features, but if clusters still collapse, the report must say so explicitly.

## 10. Why Pseudo-3D Is Not True 3D

- `Z=0` ground plane is assumed.
- Human bodies are vertical cylinders/capsules, not measured meshes.
- Without real `K`, `R`, `t` or control-point homography, depth and metric scale remain uncertain.
- The model is useful for ground-plane crowd dynamics, not for full 3D physical reconstruction.
