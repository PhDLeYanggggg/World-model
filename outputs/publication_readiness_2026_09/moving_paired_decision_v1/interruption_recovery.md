# Full-Training Interruption And Recovery

## Observed State

The original training process PID16893 last wrote a heartbeat at
2026-11-05T14:45:25Z, update1500 of the first joint-control fit. On the next
inspection the PID was absent, no matching runner existed, the terminal
session was unavailable and no completed training receipt existed. This
establishes termination, not its cause. The exit code was not observed;
no OpenMP, memory or numerical failure is inferred from a lost session.

## Recovery

The existing process lock and checkpoints were retained. The unchanged
command `train --resume` started PID18194. The next training heartbeat was
update1600 at2026-11-05T14:49:25Z. The implementation checks the full saved
binding before restoring model, optimizer, sampler, RNG, trace and update
counter. There was no code/configuration/threshold amendment or duplicate fit.
All three earlier real50+50-versus100-update recovery pilots were exact.

The full control must also exactly reproduce its prior2000-update model;
that check remains mandatory before freezing its fit receipt. Pilot replay
does not substitute for that full-run check or scientific performance.

The resumed process duration excludes the initial process and downtime.
Original2% risk/easy/support rules stay unchanged. Identity certification,
new reserved/H evaluation, Stage5C and SMC remain disabled. Training and
scientific results are reported separately in the final execution receipts.
