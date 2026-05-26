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

<!-- STAGE42_AC_REFRESH:START -->
## Stage42-AC Latest Evidence Refresh

- source: `fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-AB is now included as auxiliary-head evidence.
- Auxiliary-head conclusion: mixed / partial; not a uniform main contribution claim.
- Stage5C and SMC remain disabled.

### Newly Confirmed Failure / Limitation

- Auxiliary interaction/occupancy/physical heads are not uniformly positive: Stage42-AB shows small t50/FDE@50 support but negative all/hard ADE deltas. Treat them as partial/mixed evidence, not as a main contribution.
- Ungated neural replacement, metric/seconds-level claims, true 3D, foundation claims, Stage5C, and SMC remain rejected/not enabled.
<!-- STAGE42_AC_REFRESH:END -->
