# CREATE Read-Only Handoff Verified

## Scope and Provenance

2026-09-23. The author directed M3W to ask the existing `simulation model` task
for CREATE connection information and follow its restrictions. That task supplied
its current connection method, protected workload boundaries and dated scheduler
evidence. Its response is not authority to modify its project or submit M3W jobs.

`fresh_run`: a separate short SSH hostname query, followed by serial queue,
biomed partition and reservation queries at 10:07:54-10:08:00 UTC. All four
queries exited zero. This is an actual authenticated connection and read-only
scheduler check, not merely an SSH configuration inspection.

Reference documentation read: the supplied local runtime policy, SSH helper
lines and maintenance report. This is not a `cached_verified` scientific result.
The simulation controller was not imported or executed. Its source and ongoing
experiment were not modified.

`not_run`: M3W remote directory inventory, data transfer, environment installation,
resource reservation, job submission/cancellation/restart, training, model
evaluation or new scientific results. M3W's remote project path remains unknown;
the simulation project is explicitly not a substitute.

## Fresh Observations

| Check | Observation | Limit |
|---|---|---|
| SSH hostname | Successful authenticated login-node response | Not a GPU/runtime verification |
| User queue | Two simulation training tasks running; four array elements pending for maintenance; one CPU audit waiting on dependency | Point-in-time snapshot, not a forecast of completion |
| biomed A100 partition | State UP, MaxTime 48 hours | Does not establish free GPUs or M3W submission authority |
| Maintenance reservation | `slurm_upgrade`, all-node maintenance flag, Sep 24 07:00 through Sep 25 23:59:59 | Scheduler reservation, not a guaranteed restart/completion time |

The reservation times are scheduler local times; the simulation task's same-day
clock record is UK time (+01:00). The long-GPU restriction and prior MFA approval
expiry are supplied historical observations, not freshly rechecked here. Account
GPU concurrency, free capacity, CPU concurrency and remaining credits are not
established. In particular, eight GPUs previously observed in use are not a
verified general quota. The simulation's one-GPU/four-CPU/eight-GB request is
its workload setting, not an allocation template automatically approved for M3W.

The queue snapshot contains the protected simulation workload, not an identified
M3W run. That does not show that M3W historical assets or completed jobs are absent.
Targeted searches of M3W's local configurations, scripts and research records did
not locate a verified CREATE project path. No remote home/project tree was scanned.

## Operating Restrictions Retained

1. Use the supplied existing identity through a short separate SSH connection.
   Retain BatchMode, IdentitiesOnly, UseKeychain, 10-second connect timeout and
   strict host-key checking; bound each whole query with a 60-second parent timeout.
2. Do not read/copy/display key contents or request passwords, OTPs, tokens or
   cookies. Do not change credentials, agent, Keychain, VPN, forwarding, persistent
   SSH configuration or permissions. No ControlMaster/socket was verified.
3. Do not close other sessions, execute M3W in the simulation directory/environment,
   install packages there, clear caches, write temporary files or touch its code,
   manifests, snapshots, logs, canonical results or active/pending jobs.
4. Do not call the simulation controller's upload, submit or publish functions.
   No cancellation, requeue, walltime reduction, migration or retry job is allowed
   by this handoff. Do not use restricted long or interruptible partitions to
   work around limits.
5. Keep observations serial and bounded. Long-running workloads are normally
   checked at one-to-two-hour intervals, not by rapid polling. One bounded retry
   may follow a timeout, then back off; a timeout never proves a job has stopped.
6. Do not run scientific generation, fitting or evaluation on login nodes. This
   handoff runs no scientific computation anywhere and does not change M3W's
   research protocol or release a training job.
7. Once an explicit M3W project path is established, inspect only that authorized
   path and existing jobs before proposing new resources. Do not treat /users
   symlinks as errors to fix; use read-only resolution when necessary.

## Evidence Storage

The detailed local-only receipt is
`data/stage_cvpr2027_experiments/create_handoff_20260923/observations.json`.
It records the supplied SSH arguments, protected project/job identifiers, source
document paths and raw query responses. No secret contents were captured.
It is Git-ignored and is not published. SHA256:
`1cb4dd58246332030b4157fa860119c8dfe2158d716e2a0a98b97ed0b4b86bd6`.

This public report deliberately omits the account name, key path, internal node
inventory and protected project's detailed filesystem layout. No code changed,
so no training tests were rerun. Evidence is the four successful read-only queries,
not a unit-test or model-success claim.

## Result

The missing-connection-information blocker is resolved. The M3W remote asset path
and independent experimental data roles are still separate unresolved matters.
DroneCrowd annotation acquisition remains complete; the pending image-audit
question is not answered by this CREATE handoff. No model promotion, Stage5C,
SMC or submission-readiness claim follows from successful SSH access.
