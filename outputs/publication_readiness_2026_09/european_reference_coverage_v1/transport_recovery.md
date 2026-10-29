# Existing TRAIN Stream Recovery

29 October2026. The first support-diagnostic attempt (PID44928) exited1 after
a read-only SSH stream returned255 at12:01UTC. Four context groups had replayed
their controls, but the complete148-view report was not produced. That partial
attempt is incomplete, not a successful support result or an HPC job failure.

The error occurred while reading the already-frozen existing SOURCE TRAIN
packet for locality110 in the fifth context. No training was restarted,
submitted, cancelled or duplicated. No SSH authentication configuration is
changed. Scheduler/job state cannot be inferred from this stream error and
was not queried for this read-only diagnostic.

Before retrying transport, add a private bounded byte cache. The fetched
payload must match the original manifest's byte count and SHA256 exactly.
Use the original remote ownership/manifest checks, original unpacker and
original whole-record fitting filter. Corrupt caches fail rather than being
replaced silently. Exact report replay uses cached verified bytes without
network access. Cache cap4GiB; preserve10GiB free disk. No raw packets in Git.

Only transport/recovery changes. Model weights, forecasts, labels, splits,
eligibility, floor, oracle,148-view roster and2%requirements remain frozen.
Physical packets contain previously opened original SOURCE TRAIN rows;
the same fitting filter is applied. No independent role is opened. This does
not reinterpret those historical rows as new test data.

The initial failure remains recorded. A subsequent successful report, if any,
must be supported by its own terminal exit, complete controls and replay.

## Separate Case-Probe Implementation Check

The first new remaining-case probe (PID46941) exited1 before producing any
result. Its mean-disagreement envelope was rejected by the existing partial-
label contract, which correctly requires maximum requested-step disagreement.
The probe now calls the frozen max-envelope utility and has a regression test
with only a final-step label possibility. No original forecast readout,
148-view support result, risk budget or historical artifact was altered.
