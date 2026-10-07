# Recovery Resource Adjustment: Applied, Not Training Completion

Result source: fresh_run for scheduler/accounting observations and focused tests.
No new scientific training result, inference, bootstrap or independent evaluation.

## Evidence Sequence

1. At15:37 UTC, recovery37835856, join37835859 and diagnostic37837636 remained
   pending with zero runtime. Remote scheduler queries had occasionally exceeded
   the old20-second observation limit. SSH authentication had succeeded; this was
   not evidence of expired access or failed training. The new read-only observer
   gives queue and accounting independent60-second windows and records unknown
   state without discarding artifact observations or resubmitting jobs.
2. At15:38 UTC, comp186 had six effective CPUs and12,152MiB unallocated memory.
   Its existing four-CPU/16GiB request could not fit that snapshot. Capacity and
   future reservations may change; this is not a guaranteed scheduling window.
3. Fresh accounting verified four successful comparable batches, with maximum
   measured RSS1,651,900,416bytes, about1.54GiB. An8GiB reservation leaves about
   five times that observed peak. The memory guard requires all four records
   to remain completed0:0 and below2GiB.
4. The first update failed with the exact invalid8G-suffix error. Its immediate
   readback showed no change. The corrected version used8192MiB and required
   the pinned no-change rejection before issuing one new command.
5. The corrected command returned an unexpected-message error. Its immediate
   readback also showed the old request. It was not retried. At15:47 UTC,
   separate read-only queries for the array and its sole element both showed
   MinMemoryNode=8G, TimeMin=00:10:00, TimeLimit=12:00:00, original comp186,
   four CPUs, PENDING, zero runtime and zero restarts. Applied state is thus
   supported by delayed authoritative readback, not by command success.

The Slurm client is25.05.9. A future StartTime is a scheduler estimate, not an
actual start or commitment. A backfill allocation may receive a shorter limit
than the requested12-hour maximum. Atomic100-step checkpoints remain available;
any time-limited recovery must follow verified terminal state and resume rules.

## Scope and Verification

Fifty-one focused tests cover memory evidence, pending-state guards, rejected
syntax reconciliation, no retry after unknown/applied outcomes, independent
read-only observations and earlier backfill guards. The full historical suite
was not rerun for this scheduling-only change.

No loss, training code, data role, sampling, seed, thread count, model size,
2,000-step budget, checkpoint tolerance or scientific pass rule changed. The
110 accepted fit receipts and join are preserved. Only17 missing candidate fits
remain to train; all144 receipts must verify before readout. The separate
portability diagnostic does not establish alternate-node admissibility until
all17 controls match. Stage5C and SMC remain off; deployment stays unchanged.

## Receipts

- resilient_observation_20261007T153710.json
- zen3_resource_observation_20261007T1540Z.json (embedded time15:38 UTC)
- scheduler_resource_repair_v1.json: syntax rejection
- scheduler_resource_repair_v2.json: ambiguous command outcome
- scheduler_resource_repair_readback_20261007T1548Z.json (embedded time15:47 UTC): applied state

Filenames are filing labels; embedded timestamps define observation time.
The official [scontrol reference](https://slurm.schedmd.com/scontrol.html)
defines MinMemoryNode in mebibytes. This operational repair establishes neither
a neural improvement nor a valid safety guarantee. Coordinates and horizons
remain dataset-local/pixel and raw-frame; no metric, seconds, true3D or
foundation-model claim is added.
