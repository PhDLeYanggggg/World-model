# Current-Producer Reference-Harm Loss Comparison

Registered before training/readout, 29 October 2026. This is the direct loss-only
comparison on the current policy's producer, `european_dimensionless_refit_v1`.
The earlier-family experiment is retained separately; its lineage clarification
explains why it cannot establish a repair of the current producer.

Keep the nine single-fold fits, three seeds17/29/43, partial-neighbor geometry,
grouped agent-track Transformer, dimensionless correction fractions, causal
motion bound, 88,514 parameters, train-source normalizers, sampling, initialization,
AdamW and 4,000-update schedule. The only change is:
`mean(factor * (ADE + 2*relu(ADE-selected_ADE) + 2*relu(ADE-CV_ADE)))`.
Selected baseline is chosen on each producer's training sources. Errors require
future labels for supervision only; inference remains entirely past-only.

Before primary training, require 20-update neutral/native equality, 10+10 resume
equality, and one exact4,000-step neutral replay of the current frozen producer.
Then run all nine fits /36,000updates. Freeze final endpoints before predictions.
No checkpoint/seed/threshold selection. Original controls are cached_verified.

## Readout

All rows are already-opened European SOURCE development, not independent test.
Selection, calibration and confirmation remain unopened. Use8observations,
12requested future steps at stride12raw frames, image pixels, detector tracks.
Compare matched new and old forecasts on the same excluded sources, all18
source-excluded producer-seed contributions per locality (two producers times
three seeds per locality, with12localities overall), averaging errors, not
ensemble trajectories. Do not count repeated predictions as independent rows.
Report each seed, locality and recording; ADE/FDE, hard/easy, zeroCV, positive
harm, net degradation, completeness and worst-scene/tail behavior. Bootstrap
3,000locality draws gives conditional exploratory uncertainty, not confirmation.

Reconstruct the six frozen seed43policy contexts and exact old TRAIN partitions.
Use the original causal CV/damping.97controller floor, fixed old eligibility and
all148required recording/context views. First reproduce the old148-view oracle
diagnostic exactly. Then replace only candidate trajectories and recompute their
cost labels/envelopes. Report the formerly unsupported082/51record explicitly,
as well as any newly unsupported recording. Oracle decisions are label-aware,
not deployable. Old neural-dependent scores are not valid gates for new outputs.

The 2% full-policy all/easy harm requirements remain unchanged. Net easy error
cannot cancel positive harm. Empty/zero-support risk stays not_estimable; matched
counts are utility only. Do not certify safety from a fitted loss or source
readout. Keep deployment unchanged even if candidates improve; fresh causal cost
heads and independent calibration would still be required. No Stage5C, SMC,
metric, seconds, true3D, foundation or submission-ready claim.

Local native arm64 CPU4/interop1/workers0, heartbeat50steps, checkpoint200steps,
explicit hash-checked resume,10GiB disk reserve. No new HPC job. Weights stay local.
