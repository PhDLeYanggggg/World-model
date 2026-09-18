# Full SDD Auxiliary Input Evidence

Result source: fresh_run for full construction and independent replay;
cached_verified for source lineage and completed-cache resume. This is input
and runtime evidence, not a predictive result.

- Original train videos:40; five source scenes; 8,005,367 raw annotation rows.
- Eligible pedestrian-ego windows:229,333; eight observed/twelve requested future
  points at stride12rawframes. Requested endpoint:+144rawframes.
- Unique past frame/agent crops:254,841; private array bytes:2,612,501,343.
- Complete/partial/absent future labels:188,358 / 37,360 / 3,615.
  Membership depends only on past support. Labels and masks are separate inputs
  to the loss, never fields consumed by the prediction function.
- Boundary-partial crops:22,706; inferred dark-border affected crops:1,145;
  crops with no retained pixels:2. Explicit masks retained, no silent row removal.
- Source-frame decodes:364,516; requested RGB frames:29,387.
- Sum build/check durations:827.32seconds across40videos.
- Independent verification:all229,333query/image joins;120geometry/label replays;
 120independent source frames;944exact crop replays;600array hashes verified.
 Independent verification duration:275.40seconds.
- Future mutation checks:120, three past queries per video, all inference fields
  unchanged when future coordinates and flags change.
- Completed preparation resume:40reused,0newextractions;600array hashes and the
  complete manifest match their saved receipts. No original SDD val/test raw
  annotations or images opened by this source pipeline.

Manifest SHA256:
702b669fe4b1db254af2ec143088fa1e9be755f097696f5a69a5044de9f8e138

Registration SHA256:
17d12a071c73a44a42c42c13341c18f4cc4093188b719a4c6a0b57623e8e5fbb

## Real Training Pilot

Native arm64 CPU, Torch2.12.0, NumPy2.4.6, four Torch compute threads, one interop
thread, zero DataLoader workers. The source past-RGB seed17/fold0 trial completed
100actual optimizer steps in about5.05seconds and saved model, optimizer and RNG.
No held-fit evaluation was run by the pilot. It resumes the same registered fit,
not an extra selected candidate. A conservative all-updates-at-this-rate
extrapolation is about4.55hours; it is an estimate, not completed runtime.
Geometry is expected to cost less. The full unchanged54fit matrix is now running.

Training allows the entire229,333row source population. Each auxiliary fit makes
128,000uniform replacement draws; this is not a claim that every source row is
sampled or that a complete source epoch has run. Actual unique counts are saved.

## Boundaries

Original SDD diagnostic caches/configurations retain their old diagnostic role.
This new wrapper and registration explicitly admit only the approved train-40
source population for supervised auxiliary training. Main ETH/UCY8-to12 primary,
fit rows and sealed evaluation roles are unchanged. Three exposed fit sites are
not independent confirmation; no model is selected from their results here.

Source annotations can be retrospectively interpolated, and masks are inferred,
not human gold. This is offline-annotated prediction, not strict sensor-as-of
forecasting. SDD remains annotation-pixel/raw-frame only. No metric/seconds
equivalence, true3D, foundation, Stage5C, SMC or deployment claim.

