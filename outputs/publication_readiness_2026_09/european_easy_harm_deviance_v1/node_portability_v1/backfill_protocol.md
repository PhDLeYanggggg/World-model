# Ten-Minute Minimum Allocation for the Existing Diagnostic

This is a scheduling-only amendment for still-pending job37837636. It retains
the requested one-hour maximum, comp188, four CPU threads,8GiB memory, fixed
TRAIN inputs,17 reference identities,34,000 verification updates and exact
comparison contract. It changes only TimeMin from unset to ten minutes.
Recovery37835856 and join37835859 remain untouched; no new job is submitted.

The earlier17-pair diagnostic performed68,000 verification updates in4m10s.
This job performs34,000 updates, but that is not a guaranteed runtime estimate.
Allowing a shorter idle scheduling window may help; it does not bypass priority
or guarantee an earlier start. Slurm can reduce the allocated wall limit to a
value between this minimum and the requested maximum before starting.

If wall time is exhausted, preserve atomic checkpoint and optimizer/RNG states.
First verify scheduler terminal status and inspect logs. Only then prepare a
bounded same-job requeue or explicit recovery receipt; do not restart from step
zero or infer a crash from an observation timeout. The existing worker already
loads saved states and validates completed reference receipts.

The update guard requires the exact owned PENDING request with zero runtime and
zero restarts. An immutable intent precedes a single scheduler update. Timeout
means unknown, never success or permission to repeat. Read back actual scheduler
state before claiming that the update applied. No scientific result follows.

References: [Slurm sbatch time-min](https://slurm.schedmd.com/sbatch.html#OPT_time-min)
and [Slurm scontrol job updates](https://slurm.schedmd.com/scontrol.html).
