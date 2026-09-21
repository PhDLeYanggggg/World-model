# Native Nested Producer Launch

## Material Passport

- Mode: code experiment execution.
- Status: real training running at this recorded snapshot, not a completed cache.
- Scope: source-only fitting-lineage repair; not independent confirmation.
- Provenance: fresh_run pilot/training; cached_verified source and outer models.

Registration/code commit `2926e917` was synchronized before training. The source
preflight verified 272 bindings, 175,756 indexed queries and twelve reusable
outer models. No related M3W Python training process was found in the local
process check; disk had approximately 62 GiB available. No CREATE job or fresh
remote-queue claim was made. The preceding real 24-fit run established the
reasonableness of this local workload.

The first real pair-excluded pilot ran as PID66582/session31479 and exited 0.
It completed 100 updates in 1.9043 fit seconds, with finite loss/gradients and an
atomic checkpoint. The pilot is part of the fixed 4,000-update endpoint.

Continuation PID66606/session69788 resumed at update100, then logged150 onward.
At the 1m13s process check it was running, CPU212.8%, RSS1,763,888KiB (about1.68GiB).
A later live heartbeat/receipt check found five of18 endpoints complete and
coupa/gates/seed43 at step1800. These are timestamped observations, not proof of
eventual completion. Poll the actual session/PID plus completion receipts before
claiming completion or restarting anything.

Native arm64 Torch CPU4, interop1, workers0, checkpoint every200 updates and
heartbeat every50 updates. The total fixed budget remains72,000updates. All18
endpoints must pass before cache inference begins; no partial scoring or selection.

54 scoped producer/metric/lineage tests passed before launch. A separate training-
partition exporter has ten additional synthetic tests, including outer-payload
poisoning, improper fitted preprocessing, unknown supervision and row alignment.
Its real export is not yet run. It does not modify any frozen training dependency.

The zero-CV native-risk rule remains an author decision requested asynchronously.
This running prerequisite neither trains a risk head nor chooses thresholds,
calibrates safety, reads a closed role, deploys a model, runs Stage5C or enables SMC.
