# Stage42-BN User Action Required: Source Time/Geometry Calibration

- source: `fresh_source_time_geometry_calibration_audit`

## Before Any Metric / Seconds-Level Paper Claim

- Confirm source-specific coordinate convention and homography direction for each ETH/UCY source used in a calibrated subset.
- Confirm whether model horizons are annotation-step horizons or raw video frame horizons for the exact feature/evaluation rows.
- Keep SDD as pixel raw-frame unless a source-specific scale and annotation-stride protocol is explicitly validated.
- Do not use TrajNet snippets for raw-frame t100 or metric/time claims.
- Do not use TGSIM traffic metric evidence as pedestrian world-model success.

## Current Allowed Wording

- Allowed: source-specific ETH/UCY calibration evidence exists for selected sources.
- Not allowed: M3W is metric, seconds-level, true 3D, or foundation-scale.
