# Fit-Only Static-Scene and Start-Direction Probe

Prospective diagnostic after the completed stationary-start pooling study.
The parent eight-observed/twelve-predicted-step task, primary metric, fit folds,
development rules and raw50 supplement remain unchanged. This is not a new
independent benchmark or a policy-selection run.

## Hypotheses

The prior pooled neighbor classifier has one-way, not bidirectional, probability
lift. Test whether static obstacle context supplies missing start/direction cues.
Probability improvement alone does not establish trajectory improvement.

ETH/Hotel have supplied static obstacle XML files, H matrices and reference
images. The images visibly include people and their capture times are unverified;
they are audited for availability only, never encoded as features. Supplied
destinations and groups are not used: their entire-recording annotation provenance
does not establish past-only goals or group membership. No video frames are used.
The XML is an unverified static-scene proxy, not human-gold semantics, a ground
truth walkability map, a physical safety rule or newly approved visual modality.
No third-party asset is redistributed. Record its hash and geometric limitations.

## Fixed Experiment

- Reuse exactly the 365 source-verified stationary fit rows and their physical
  scene folds. No new rows, longer history, development or confirmation labels.
- A nearest supplied obstacle defines the query's normal/tangent coordinate
  frame. Translation, rotation and positive coordinate-scale changes must not
  change relative features. A degenerate/ambiguous frame is explicitly flagged;
  a directionless frame gives zero directional input and target support.
- Scene normalization uses the static obstacle extent, not target displacement
  or held-trajectory statistics. It changes this diagnostic regression target
  representation, not the parent forecasting metric. Restore predictions to
  native coordinates and score with the original past normalizer as well.
- Compare three feature sets: pooled neighbor control; control plus static map
  distances/shape context; control plus static map and directional neighbor
  summaries. No scene ID, absolute position/frame, whole-run length, future
  availability, supplied velocities, teacher or endpoint inputs.
- Fit the same fixed logistic/ExtraTrees start classifiers (C=1; 256 trees,
  minimum leaf10) and Ridge(alpha=1)/ExtraTrees multi-output trajectory regressors
  (same tree settings), seeds17/29/43. Every setting is reported. Two physical
  held folds x3seeds x3feature sets x2model families x2tasks =72 fitted models.
- Classification baseline: smoothed train-only prior. Regression controls:
  zero-displacement CV and opposite-scene train-only mean relative trajectory.
  Train all 12-step trajectories including zero targets, never positives only.
- Report unrestricted regressors and a fixed p(change)>=0.9 diagnostic gate
  using the paired classifier. No threshold search or deployment selection.
- Probability: Brier/AUC and equal-agent/run Brier. Trajectory: all stationary
  rows' native/past-normalized ADE/FDE, CV-relative harm, unchanged parent easy
  definition, endpoint direction error where both direction vectors exist.
  Report zero-direction exclusions. Conditional angular error cannot replace
  all-row trajectory error. No pooled raw-coordinate cross-dataset score.
- Fresh source/preprocessing checks, cached row identity checks, per-model
  checkpoints, resume hashes and heartbeat. A small local CPU experiment is
  adequate; no new GPU/HPC allocation is justified by this diagnostic size.

## Interpretation Rules

This is adaptive fit-only exploration, not a confirmation exercise. Repeated
seeds and overlapping windows are not independent sites. No significance claim
or narrow window-bootstrap CI will be made. A result must work in both directions
and retain easy cases before motivating a start-aware neural forecast. Do not
interpret successful extraction, reduced latent distance, or useful start AUC
as neural forecasting lift. Preserve local positives and failures alike.

No parent checkpoint/source hashes are overwritten. No change to the primary
metric, no dropping stationary labels, no test-set calibration, no Stage5C/SMC,
no metric/seconds, true-3D, foundation or submission-ready claim.
