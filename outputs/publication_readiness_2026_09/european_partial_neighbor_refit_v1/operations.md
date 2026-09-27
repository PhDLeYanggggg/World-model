# Execution and Resource Record

Native arm64 Torch CPU, four compute threads, one interop thread, zero workers.
No NumPy fallback, resource probing, MPS probe or DataLoader multiprocessing.

| Phase | PID | Process seconds | Peak RSS (GB, decimal) |
|---|---:|---:|---:|
| pilot | 3398 | 4.70 | 0.713 |
| control | 3422 | 92.10 | 1.064 |
| train | 3599 | 902.94 | 1.655 |
| predict | 4910 | 669.15 | 2.332 |
| evaluate | 6080 | 6.20 | 1.249 |
| replay | 6224 | 671.90 | 2.364 |
| verify_eval | 7344 | 6.08 | 1.230 |

Nine new fits: 36,000 updates, 899.93 training seconds.
Fresh control: 4,000 updates, 91.87 cumulative training seconds.
The100-update pilot resumes into the same4000-update control and is not counted twice.
All final sampling counts and generator states match their registered controls; held fitting draws are zero.
Model parameters and control resume match exactly. Checkpoints retain optimizer/random state.

CREATE queue inspection succeeded read-only; no remote job was submitted, cancelled or modified.
The local pilot fit the resource envelope, so no remote migration was needed. The preflight
preserves10GiB free space. Slow completed phases are not downgraded or relabeled.

Registration c2df9fb5 precedes training. Prediction freeze39607e9e precedes scoring.
Result commit27eea024 retains the failed benefit screen before complete replay sealing.
Private cache/checkpoints/predictions/logs remain ignored; public files contain code,
configuration, hashes and aggregate evidence only. Unrelated staged work is not committed.

Source arrays/legacy weights: cached_verified. Geometry, new weights, source readout and
fresh prediction replay: fresh_run. Independent confirmation and cold raw-download rebuild:
not_run. The full historical test suite is not_run; scoped verification has its own receipt.
No deployment change, Stage5C execution, SMC, metric or calibrated-seconds claim.
