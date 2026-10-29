# Verification and Completion Record

Verified2026-10-29 after the complete readout. This is verification of research
artifacts, not independent scientific confirmation.

- Real training: PID42524 exit0;54/54 new fits,108,000 updates,526.972002s.
- Real inference: PID43528 exit0;144/144 fixed fit/held views,118.259964s.
- Frozen hashes and complete model/source/seed rosters verified.
- All three cached control policies and signed MSE reproduce exactly.
- Three-arm initialization and continued sampler states match within each fit.
- Protected B/R/Re outputs match the frozen anchor on every inference row.
- Inference repeats exactly; outcome labels do not filter causal decisions.
- Same-anchor cohort identities, matched counts, recording partitions and
  risk/completion arithmetic verified.74 fit and31 held pairs are disjoint.
- Reports regenerate byte-for-byte from frozen evidence.
-83 scoped tests pass in4.78s on the final tested implementation. Coverage includes
  numerical failure/repair, exact resume, protected weights, causal inference,
  partial/duplicate rosters, corrupted records and failed-risk gate behavior.
- Full legacy tests not_run: some write into frozen research outputs. A cold
  raw-data rebuild and independent confirmation also remain not_run.
- Both private scientific figures visually checked: readable labels, no clipping,
  negative intervals and exposed-TRAIN limitation retained. Figures are not in Git.
- No listed training/pilot/readout PID remains active. Final heartbeats are history.
- No CREATE scheduler job was submitted; existing hash-verified TRAIN packets were
  streamed to native local arm64 CPU4/interop1/workers0 execution.

The precursor PID40764 exited1 after37 completed fits. Its failed numerical
path and artifacts remain separately recorded; no precursor result is counted
as a completed repaired fit. The numerical diagnostic continuation is likewise
engineering-only and excluded from the54 finals.

Registration SHA256:
`24decd67e7dab5144d9e0731b245f1b884e341c0b6fddac7045ccfaf17053291`

Training freeze SHA256:
`740eae7041ac2c301eab9da67f883fcb5d9d0e8101996e8d52f3a0a25553ecde`

Readout summary SHA256:
`dedc2ed5e8b913db18b733774c0582b64ec814b907ef107bfbd9d185518c325d`

No deployment or reserved data role changed. Stage5C/SMC not executed.
Registered advancement failed; M3W submission readiness remains unproven.
