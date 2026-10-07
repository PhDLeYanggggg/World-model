# Match the Existing Recovery Reservation to Measured Memory

At15:38 UTC on7 October, comp186 had six effective free CPUs but12,152MiB
unallocated memory. Recovery37835856 still requested four CPUs and16GiB. The
earlier CPU-only queue bottleneck had changed: its memory reservation now also
prevented fitting into the observed space. Availability is transient, not a
guarantee of admission.

Before mutation, re-read four completed comparable batches through accounting.
Require all four to have successful terminal records and measured peak RSS
strictly positive and no greater than2GiB. Only then reduce this same pending
recovery's memory request from16GiB to8GiB. This keeps at least four times the
admitted measured peak; it is resource sizing, not a smaller model or dataset.

Also permit a minimum ten-minute backfill allocation, keeping the12-hour
requested maximum. The previously completed comparable tasks took at most
9m51s for larger fit blocks; this is evidence for trying backfill, not a runtime
guarantee. Atomic100-step optimizer/RNG checkpoints remain in place. An expired
allocation requires authoritative terminal-state inspection and checkpoint-based
recovery; no automatic duplicate submission or reset to step zero is allowed.

Everything scientific stays fixed: comp186, four compute threads, one interop
thread, zero workers,17 remaining candidates,2,000 updates each, original input
roles, losses, seeds, sampling, exact reference checks and all144-fit admission.
All110 accepted receipts and join37835859 are preserved. No node switch is
authorized by this update, so the separate comp188 verification is not used as
an admission substitute.

The guard validates owner, registration and execution-amendment hashes, all
preserved references, exact PENDING resource fields, zero runtime and zero
restarts. An immutable intent precedes one update. A command timeout remains
unknown until read-only reconciliation; do not retry automatically. A denied
update, remaining queue wait or later OOM is reported, not called completion.

Only MinMemoryNode and TimeMin are changed.
[Slurm job-update documentation](https://slurm.schedmd.com/scontrol.html).
No training result or deployment improvement is established by this amendment.

## Confirmed Syntax Rejection and Corrected Attempt

The first command returned1 at15:43 UTC: this installed scontrol rejects the
8G suffix for MinMemoryNode. The immediate authoritative readback preserved
16G/one-hour minimum/PENDING/zero runtime. The immutable v1 receipt records this
failed attempt. The official scontrol reference specifies mebibytes; the
corrected command uses MinMemoryNode=8192 with exactly the same resource goal.

Version2 requires the pinned v1 receipt to contain this exact syntax error,
returncode1 and unchanged before/after fields. It still rechecks current state,
registration, references and memory evidence. Unknown outcomes, other errors,
successful or partially applied changes cannot authorize this correction.
A separate immutable v2 intent and receipt preserve both attempts. No automatic
retry after any v2 outcome is allowed.
