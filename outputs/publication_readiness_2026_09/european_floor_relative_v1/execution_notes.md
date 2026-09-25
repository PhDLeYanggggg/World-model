# Execution and Verification

## Provenance

Registration commit: `1049876b`, pushed before the pilot and new readout.
Preceding diagnostic: `ed981515`; bound summary SHA-256:
`e432f863c2bb6164b495a7d808a7f115179cb3d75e9d85221cb39cb9d058bc5b`.
The configuration, runner, core implementation, initial tests and registration
are immutable bindings of every training identity. Reports are derived after
training, not inputs to it. New checkpoints/decisions/evaluations are fresh_run;
predecessor forecasts and population are cached_verified. Independent reserved
selection/calibration/confirmation is not_run. No role was relabeled.

## Runtime

Native arm64 Python 3.11.1, Torch 2.12.0, NumPy 2.4.6, CPU computation threads 4,
inter-op 1, DataLoader workers 0. The entrypoint's architecture guard executes
before Torch import. The 100-update real-training pilot resumed to 2,000 in the
same model, with optimizer and sampler state. This is real Torch training, not
a NumPy fallback. There was no observed runtime crash, hang or failed phase.

| Phase | PID | UTC start | UTC completion |
|---|---:|---|---|
| Pilot, batch | 65782 | 13:18:51 | 13:18:55 |
| Inner floors, batch | 65834 | 13:19:37 | 13:22:06 |
| Inner floors, fitting | 66108 | 13:22:56 | 13:24:55 |
| Main heads, batch | 66389 | 13:26:51 | 13:31:25 |
| Main heads, fitting | 66861 | 13:32:27 | 13:37:24 |
| Decisions, batch | 67376 | 13:38:46 | 13:39:35 |
| Decisions, fitting | 67491 | 13:40:06 | 13:40:56 |
| Evaluation, batch | 67575 | 13:41:16 | 13:43:55 |
| Evaluation, fitting | 67806 | 13:44:17 | 13:46:55 |

All timestamps are 2026-09-25 and all listed processes terminated successfully.
Both decision banks preceded either evaluation. Wall times include assembly,
hash checks and inference; summed optimizer-fit time is 582.284 seconds,
including the pilot, not the full experiment elapsed time. A process observation
showed roughly 6.9 GiB RSS; this was a sample, not a measured peak. Disk free space
was about 24 GiB after the fits; a 10 GiB stop-and-preserve guard remains active.

CREATE was queried read-only at 13:18:17-13:18:20 UTC. It returned normally with
six unrelated visible scheduler rows, five pending and one running, and no
visible M3W row. No job was submitted, cancelled or modified. This timestamped
queue observation is not an inspection of remote result folders or a claim
that remote assets are absent. The local receipt remains private, SHA-256:
`b78574136c1c676ef9f0ba8958cb4b043d90f22e968d537215305b88d9c87f8a`.

## Checks

Every new head completed 2,000 updates, giving 234 heads and 468,000 updates.
The pilot contributes 100 updates and its resume 1,900, not 2,100. The first
up-to-4,096 saved inference rows replay exactly from every checkpoint; full
score banks are hash-bound. This is a prefix replay, not a claim of full-array
inference replay. Source-balanced draws and fitting-only preprocessing agree
within all 36 main comparison groups. Unknown-label supervised draws: zero.

Independent calculations verify 144 causal decision arrays, 144 coordinate
error arrays and 2,160 metric reductions. The frozen preceding control matches
180 metrics exactly. Every group retains all four new policies and the old
control. The statistical unit is locality; repeated seeds/folds/windows are
not counted as independent samples. The support/stop-start analysis is explicitly
post-hoc, uses no new policy and did not change training or decisions.

301 scoped tests in 47 files pass in the recorded final test run. The full
legacy suite was not run. The private test receipt binds its exact file list
and log; public `completion_checks.json` links it with the aggregate files.
Both generated figures were visually inspected for legible labels and axes.

## Storage and Continuation

Only code, configuration, reports, SVG charts and lightweight aggregate tables
are eligible for Git. Checkpoints, score banks, decision arrays, logs, image
renders and data remain private. The 3,019 pre-existing unrelated staged entries
must remain untouched. Their raw staged-diff checksum before this task's result
commit is `c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.

Resume commands are in `operation_zh.md`. A completed phase is not an instruction
to restart training. A changed binding requires a separately registered
experiment, not overwriting this evidence. Deployment remains unchanged.
No Stage5C execution, SMC, metric/seconds, human-gold, physical-safety, true3D,
foundation or independent-confirmation claim is made.
