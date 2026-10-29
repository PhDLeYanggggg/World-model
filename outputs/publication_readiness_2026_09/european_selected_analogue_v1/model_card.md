# Model Card: Recording-Excluded Analogue Information Probe

- Purpose: diagnose whether existing causal features contain selected-cohort
  harm information beyond source averages and recording-preserving permutation.
- Status: completed fitting diagnostic; screen unpassed because full support
  is unavailable. Not deployable, no new neural checkpoint or optimizer updates.
- Input: existing380 causal features, causal disagreement envelope, original
  causal selection, recording key. Target labels never enter query features.
- Supervision: paired harm/easy-harm fractions from known other-recording fitting
  donors. Unknown query outcomes stay in inference. No held/test reference labels.
- Retrieval: exactly32 Euclidean neighbors, donor-only standardization, std
  floor1e-6, clip8; no tuning. Same-record donors excluded. Unsupported queries
  remain NaN and cannot be silently interpreted as safe.
- Controls: donor source mean, same neighbors with paired labels shuffled within
  each donor recording; original and estimation-only frozen neural readouts.
- Dataset roles: exposed original TRAIN, inner-fit only. Upstream training
  exposure means reference exclusion is not whole-pipeline out-of-fold learning.
- Evidence:72 views,24 fully supported,48 unsupported. Local matching signal
  exists descriptively, but magnitude estimation still fails. No independent CI,
  policy utility, ADE/FDE lift, generalization or deployment improvement claimed.
- Limitations: sparse source recording coverage, donor/query selection shift,
  repeated occurrences, detector-silver outcomes, one fixed permutation seed,
  high-dimensional fixed-distance matching, copied benefit/reference outputs.
- Output restrictions: do not use as a probability-calibrated harm detector,
  safety guarantee or controller. Deployment is unchanged.
- Units: pixel/local, raw observation stride12, obs8/pred12. No metric/seconds,
  human gold, physical safety, true3D or foundation-model status.
- Prohibited stages: Stage5C execution false; SMC false.

See `registration.json`, `complete.json`, `summary.json`, `verification.json`
and the reproduction instructions for exact lineage and verification scope.
