# Execution And Verification Receipt

## Fresh Execution

| Phase | PID | Duration seconds | Peak RSS bytes | Outcome |
|---|---:|---:|---:|---|
| Metadata preparation | 46564 | 17.698035 | 1199865856 | Completion receipt present |
| Real pilot | 46811 | 0.742392 | 706412544 | Observed exit 0 |
| Full retrieval | 46854 | 37.138408 | 1521188864 | Observed exit 0 |
| Complete recomputation | 46936 | 32.253634 | 1205682176 | Observed exit 0 |
| Read-only saved-reference replay | 47266 | 2.608450 | 1035943936 | Observed exit 0 |

Recomputation here means a new execution of the same eight source-held
retrievals with chunk31 rather than chunk64. It is not independent scientific
confirmation, a new data split or a new seed. All reference IDs, distances,
predictions and metrics compare exactly. No same-source or same-recording
references are used. All banks have at least32 identity buckets.

During replay, fitting and distance-search functions are replaced with errors.
Saved references reconstruct all predictions and metrics exactly. All24 public
files existing at replay start retain identical hashes afterward. This receipt
is added afterward, not falsely counted among those24. No concurrent report
writer ran during the read-only interval.

The analysis SHA256 is
`04039e4f9fb74fc4f38112e3166ce2b06f2dbb952730c96250838ae4cde98a66`.

## Tests

Eighteen focused tests pass in2.17seconds, covering source exclusion,
target-perturbation invariance, distinct buckets, chunk equivalence,
unsupported banks, invalid inputs, deterministic null assignment, the existing
boundary/channel interfaces and reporting. Lower MSE/cost cannot hide higher
harm in the screen; a zero reference remains NA. Full historical suite was
not rerun. These tests do not establish the research hypothesis.

## Artifact Boundaries

Native arm64 local execution; CPU4/inter-op1/workers0, no resource probing.
The inherited forest_threads1 banner is not a new forest-training claim.
Source features and controls are cached_verified; metadata and retrieval
predictions are fresh_run. New neural training, new C/H, independent test,
full-policy safety and deployment evaluation are not_run. No new CREATE job.

The main manuscript PDF is unchanged with SHA256
`d9babe3ddbac858bcc1ac30e3ba98b332bed625c78ff9231d1d42a3e30b11772`.
The new paper addendum is separate from that compiled PDF.

Original2% rules, support, empty-control NA and original24-group primary NA
remain. No identity certification, Stage5C, SMC or submission readiness.
