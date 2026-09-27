# Execution and Resource Record

Native arm64 Torch CPU, four compute threads, one interop thread, zero workers.
No NumPy fallback, DataLoader multiprocessing or hardware-resource probing.

| Phase | PID | Process seconds | Peak RSS (GB, decimal) |
|---|---:|---:|---:|
| pilot | 8236 | 4.21 | 0.586 |
| train | 8286 | 592.13 | 1.621 |
| predict | 9269 | 576.87 | 1.670 |
| evaluate | 10194 | 9.77 | 2.876 |
| replay | 10420 | 559.73 | 1.699 |
| verify_eval | 11479 | 9.26 | 2.824 |

Nine new fits: 36,000 updates, 590.28 cumulative fit seconds.
The 100-update pilot resumes into the first endpoint; it is not counted twice.
All paired sampler states/counts and initial parameters match. Held fitting draws are zero.
Checkpoints retain optimizer and random states; original controls are untouched.
CREATE was checked read-only, no remote job submitted, cancelled or modified.
The local pilot fit the resource envelope; no remote migration was necessary.
Checkpoints, private predictions, raw data and detailed logs are not committed.

Existing geometry, control checkpoints and controls: cached_verified, with fresh control inference.
New training, predictions, scoring and replay: fresh_run.
Independent confirmation, full historical test suite and cold raw-download rebuild: not_run.
No new risk policy, deployment change, Stage5C execution or SMC.
