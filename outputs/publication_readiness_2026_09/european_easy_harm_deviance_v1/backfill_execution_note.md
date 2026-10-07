# Pending-job backfill allowance

At11:41 UTC on7 October the existing recovery array37835856 remained PENDING
with Reason=Priority and no restarts. The scheduler estimated a9 October start;
that is a changeable estimate, not a promise. The required diagnostic node stays
erc-hpc-comp186. The prior completed36-fit shards took5m17s,7m20s and9m51s; the
68,000-update verification diagnostic took4m10s. The remaining job needs only17
new2,000-update candidate fits, plus existing-state adoption and verification.

Set `TimeMin=01:00:00` on this same pending job while retaining its submitted
12-hour requested maximum. Slurm may lower the allocated time limit, but not below
the minimum, to exploit a backfill window. This changes scheduling admission, not
the training budget. It creates no new job, changes no checkpoint reference or
node and does not read development outcomes. If the allocated wall time proves
insufficient, preserve atomic checkpoints and recover the unchanged full budget;
do not report a partial fit as complete. The existing verification dependency
continues to guard the readout.

[Slurm sbatch documentation](https://slurm.schedmd.com/sbatch.html) describes
the minimum-time backfill behaviour; [scontrol](https://slurm.schedmd.com/scontrol.html)
documents the pending-job update. Faster queue admission is not guaranteed.

The first authenticated update attempt timed out after20seconds. A subsequent
read-only scheduler response confirmed the same pending job, zero runtime,
`TimeMin=N/A` and no update receipt. This is not a training failure. The bounded
retry uses a60-second command timeout and an immutable intent before the update;
an ambiguous result is recorded and must be inspected, never automatically retried.
The training job and numerical contract remain unchanged.

At11:48:15 UTC, a later read-only response showed `TimeMin=01:00:00` on the same
pending job, still with a12-hour maximum and zero runtime/restarts. The original
timed-out request therefore took effect after the earlier observation. The retry
was rejected by its pre-update state guard; it did not write an intent or issue
a second update. The original command's exit status remains unknown. The separate
`backfill_after_timeout_observation.json` records the observed applied state,
not a fabricated successful command receipt. No more scheduler update is needed.
