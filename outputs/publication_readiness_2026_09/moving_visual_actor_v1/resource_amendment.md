# Resource-Only Resume After A Disk Guard Stop

At 2026-11-06T14:45:54Z, training PID28399 exited1 because filesystem free space
fell below the original engineering reserve5GiB. Five full treatment fits were
complete; the sixth reached update3551. Its last atomic checkpoint is3400,
not3551. No donor predictions or outcomes had been scored. The current
experiment occupied211,413,472bytes; available filesystem space was4,232,028,160
bytes at the recovery check. Swap usage was zero. The external cause of the
filesystem-space change is not established.

The original registered code, config, training identity, model, samplers,
losses, preprocessing,4000update endpoint,12hphase boundary and1GiBnew-private
artifact cap are unchanged. The resource wrapper changes only the filesystem
reserve from5GiB to2GiB. This leaves more than twice the entire remaining
artifact allowance available before fitting restarts. It does not alter the
original2%all/easy risk, signed easy-net, reference/support/NA rules or introduce
an identity certificate. No existing data, caches or checkpoints are deleted.

The wrapper and amendment are committed before resuming. Existing complete
receipts are verified and reused; the incomplete motion model resumes its
model, optimizer, sampler and RNG from3400. The151updates after that checkpoint
are repeated, not counted as additional scientific training budget. Actual
compute therefore includes151discarded optimizer updates beyond the originally
planned60,200. A completed exact recovery pilot already verifies scientific
state preservation under the same checkpoint mechanism. The two complete
registered treatment refits remain required.

This amendment is engineering recovery, not evidence of predictive benefit.
If the2GiBreserve or the unchanged artifact cap is breached, training stops
again rather than silently reducing the experiment or deleting unrelated data.

```sh
.venv-pytorch/bin/python scripts/resume_m3w_moving_visual_actor_resources.py train --resume
```
