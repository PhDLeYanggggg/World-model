# Alternate-Node Exact TRAIN Replay

At12:49 UTC on7 October, the pinned recovery node comp186 had124 effective CPUs
and124 allocated CPUs; the other four physical cores were system-reserved, not
free compute. Memory was not the bottleneck. The mutable scheduler estimate had
moved to11 October. Completed comparable tasks peaked between0.56 and1.54GiB RSS.
A same-feature zen3 node comp188 showed spare schedulable CPUs.

Preserve pending training37835856 and join37835859. Submit one isolated,
TRAIN-only portability diagnostic on comp188: four compute threads, one interop
thread, zero workers,8GiB memory and one-hour wall limit. This is not a duplicate
candidate-training job. It runs the unchanged quadratic trainer for the17 frozen
remaining identities at2,000 updates each:34,000 verification updates.

Compare all original control fields against the existing immutable original-
trainer reference for each identity: model, optimizer, initial state,
preprocessing, settings, inputs, seed, update count, Torch/sampler RNG, draw hash,
row draws and query count. Identity also must match. No tolerance or reference
exception is added. All17 must match exactly before considering a scheduling-only
move of the pending recovery. A mismatch is reported, not repaired by replacing
the reference. Existing110 accepted receipts and all scientific gates stay frozen.

Separate directories, immutable receipts, atomic checkpoints, persistent
heartbeat and a process lock isolate the probe. The existing checkpoint-write
lock and combined256MiB cap/10GiB reserve remain enforced. The diagnostic adds at
most17 small state files; submission requires34MiB spare inside the existing cap.
No original checkpoint, raw packet or other project is modified. Interrupted
work resumes the same saved state after authoritative job-state checks.

The collector retains a complete negative result as well as a positive one.
Exact replay only establishes this execution check; it is not prediction lift,
independent confirmation, or automatic authorization to deploy the candidate.
No validation/independent roles are read. No Stage5C or SMC execution occurs.
