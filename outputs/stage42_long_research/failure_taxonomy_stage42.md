# Stage42 Failure Taxonomy

## Confirmed Failures / Negative Evidence

- Ungated endpoint/full-waypoint neural is not deployable: easy degradation remains unsafe.
- Internal self-gate, uncertainty gate, harm gate, and conformal-risk gate can produce large raw lift but violate proximity/collision safety in the fresh Stage42-E study.
- JEPA-only and JEPA+Transformer hybrid attempts remain negative or fallback-only in cached-verified same-protocol architecture evidence.
- Full Stage42-D retraining of every named component has not been completed; Stage42-D is an evidence audit with fresh safety/waypoint rows plus cached-verified Stage30/41 component evidence.
- Metric/time claims remain blocked by missing verified homography/FPS/stride calibration for the pedestrian external benchmark.

## Root Causes

- Safety/floor dependence: neural proposals can help hard/long-horizon slices but are too risky without the teacher floor.
- Calibration gap: dataset-local and raw-frame coordinates prevent metric/seconds claims.
- Evidence gap: some contribution claims rely on cached-verified prior ablations rather than fresh Stage42 retraining.
- Data gap: broader legally verified top-down external domains remain needed for stronger generalization claims.

## Current Best Safe Action

Keep `current_composite_tail_policy` as the deployable policy. Do not execute Stage5C or SMC.
