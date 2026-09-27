# Model and Data Card: Frozen Score Calibration

- Models: nine frozen dimensionless forecasters; matched moment and signed-score
  heads; fixed damping; existing utility/easy heads. No new training updates.
- Data:318969 source-development histories from12 previously opened European
  localities. Inputs observe8 steps and predict12 at raw-frame stride12.
- Provenance: detector-derived silver, image-local coordinates. Not human gold,
  metric, seconds, physical safety, true3D or foundation evidence.
- Roles: producer4, controller4, outer4. Every outer view uses2 calibration and
  2 held sources, both excluded from all model/preprocessing fitting. Six
  rotations per group, three seeds. Global independent roles stay closed.
- Causal inputs:355 history/neighbor/rollout features; last-step movement;
  fitting-only normalization/support; frozen utility/easy and risk scores.
  Future labels/masks/costs cannot enter action construction. Calibration labels
  are sliced by role before threshold selection.
- Objective families: moment-loss and signed-risk-loss checkpoints are equally
  ensembled, not selected using this readout. Signed components are not moments.
- Controls: guarded, supported, calibrated, calibrated_supported, legacy full
  point policy. Neural and damping receive matching role/seed/safety treatment.
- Calibration: empirical finite-grid, per-calibration-source constraints. No
  uncertainty correction or conformal/physical-safety certificate.
- Result: primary neural-vs-damping benefit fails; observed easy preservation
  improves, but selected-set risk violations remain. Do not deploy a new model.
- Limitations: development exposure, correlated views, overlapping windows,
  only two calibration sources per view, generic diagonal support heuristic,
  partial future label availability and detector noise. No new scene-joint test.
- Intended use: reproduce the controlled research comparison, not autonomous
  navigation, real-world physical safety or benchmark leadership claims.
- Stage5C and SMC: disabled. Full legacy suite/cold raw rebuild: not_run.
