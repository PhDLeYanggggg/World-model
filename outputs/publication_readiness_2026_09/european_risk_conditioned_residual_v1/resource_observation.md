# Resource And Resume Evidence

The fitting inputs and frozen checkpoints are local. This experiment fits
small fixed ridge probes and performs CPU inference; no new GPU training is
required. Native arm64 Python, CPU threads 4, interop threads 1 and workers 0
are configured by the runner. The actual fitting/prediction work completed,
not merely a Torch import. Original neural checkpoints remain unchanged.

Fitting completed under PID 67898 and the source readout under PID 68528.
On continuation, the old readout handle was missing; the OS confirmed that
PID 68528 was absent. Its complete receipt listed all 36 groups and 1,728 MSE
checks, and the last event recorded group 36. The readout was not restarted
merely because its original observation handle had expired. The subsequent
explicit replay verifies completed artifacts rather than creating new trials.

Read-only CREATE queue receipts:

- Initial: 2026-09-26 07:16:32-07:16:33 UTC, return code 0.
  SHA-256 `e4ef5ac677cd860833fa18e871fd5f088711c6162f92997cf8b2a527e3fabff2`.
- Continuation: 2026-09-26 19:41:44-19:41:46 UTC, return code 0,
  no rows in the authorized query at that time.
  SHA-256 `d15c0be021bfa41f9686b3c84500300eb745f9058a44f3e67bfef1f00290dc3e`.

No remote jobs were submitted, changed or cancelled. An empty current queue
query is not evidence of how earlier unrelated jobs ended. Full SSH receipts
remain in the ignored private experiment directory, not GitHub.

The runner preserves a 10 GiB disk reserve and per-view receipts. Summed
per-view fit/inference was 57.873162 seconds; that excludes loading, preflight,
analysis and replay. The elapsed gap between execution and continuation must
not be reported as training compute. No raw data, score caches, checkpoints,
third-party images/videos or unrelated staged files are included in this work.
