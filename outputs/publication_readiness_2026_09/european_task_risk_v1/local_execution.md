# Local Execution After CREATE Quota Preflight

The repaired CREATE preflight measured50,000,000,000bytes quota and
45,572,776,137bytes used, leaving4,427,223,863bytes. This is below the registered
10GiB reserve. It stopped before any submission intent or job. No quota was
increased, unrelated files removed, or safety reserve weakened.

The Mac has approximately24GiB free. Use native arm64 `.venv-pytorch/bin/python`,
four compute threads and zero loader workers. Stream one hash-verified source
TRAIN packet at a time from its existing CREATE location, unpack in memory,
and do not maintain a packet disk cache. Keep10GiB free and the256MiB checkpoint
cap. Computation stays local; the remote side only reads registered TRAIN files.

Run the same500-update real-data pilot, then all216 fixed2,000-update fits if
the pilot's measured runtime including packet transfer estimates below12hours
with1.5x margin. The local memory guard is36GiB on the48GiB machine. An exceeded
budget is not permission to shrink the scientific target. Preserve checkpoints
and diagnose/resume. No formal CREATE training was run in this experiment.

All numerical functions, input packets, random seeds, initialization, query
sampling, objective weights, risk definition and readout gates are unchanged.
Original-control exactness is checked in the actual local environment. No
cross-platform bitwise historical reproduction is claimed. Local source and
checkpoint receipts name the execution amendment explicitly.

Commands:

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_task_risk_local pilot
.venv-pytorch/bin/python -m scripts.run_m3w_task_risk_local train
# Only after an interruption, with the original checkpoint files retained:
.venv-pytorch/bin/python -m scripts.run_m3w_task_risk_local train --resume
```

This changes execution placement only. Independent roles remain closed;
exposed development is not confirmation, and training completion is not lift.
