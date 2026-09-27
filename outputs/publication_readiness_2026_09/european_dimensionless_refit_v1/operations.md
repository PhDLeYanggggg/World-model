# Execution and Resource Record

Native arm64 Torch CPU, four compute threads, one interop thread, zero workers.
No NumPy fallback, DataLoader multiprocessing or hardware-resource probing.

| Phase | PID | Process seconds | Peak RSS (GB, decimal) |
|---|---:|---:|---:|
| pilot | 12728 | 3.98 | 0.578 |
| train | 12768 | 553.82 | 1.635 |
| predict | 13664 | 466.64 | 1.636 |
| evaluate | 14333 | 9.77 | 2.837 |
| replay | 14361 | 452.44 | 1.636 |
| verify_eval | 14984 | 9.66 | 2.835 |

Nine new fits: 36,000 updates, 551.74 cumulative fit seconds.
The 100-update pilot resumes into the first endpoint; it is not counted twice.
All paired sampler states/counts and initial parameters match. Held fitting draws are zero.
Checkpoints retain optimizer and random states; original controls are untouched.
CREATE was checked read-only, no remote job submitted, cancelled or modified.
The local pilot fit the resource envelope; no remote migration was necessary.
Checkpoints, private predictions, raw data and detailed logs are not committed.

Separate fixed-first training replay: 4,000 updates, 62.58 process seconds, peak RSS 1.109GB.
Weights, optimizer, logged losses and sampler/RNG states reproduce exactly.
This is not an extra candidate or independent replication. Total work including verification:40,000 updates.
The replay overlapped prediction generation, so timings are operational receipts, not isolated speed benchmarks.
Its child PID was not recorded by that short verification runner; the nine-model main run has PID/heartbeat logs.

Existing geometry, control checkpoints and controls: cached_verified, with fresh control inference.
New training, predictions, scoring and replay: fresh_run.
Independent confirmation, full historical test suite and cold raw-download rebuild: not_run.
No new risk policy, deployment change, Stage5C execution or SMC.

Verified runtime: Python 3.11.1, Torch 2.12.0, NumPy 2.4.6, Darwin arm64.
These versions accompany actual training and exact inference replay, not import-only validation.
