# Complete Fixed-Budget Neural Cost Training

## Material Passport

fresh_run:all18 Torch critics,36,000primary updates and2,000extra exact first-head refit.
cached_verified:18 matched regression controls,original forecasts/normalizers/causal features.
not_run in this training receipt:complete policy outcomes and independent confirmation.

PID76819 completes2026-11-01T15:26:49Z,exit0. Timed phase431.611798s,
peakRSS10,746,478,592B;admission before runtime_ready is excluded.
All18heads finish2,000updates with30,857parameters;matched input hashes,initialization,
query draw streams,reference moments and update budgets agree with cached controls.
The first complete2,000-update head independently refits exactly,including model,
optimizer,sampling states,duals and all non-timing scientific trace fields.
Checkpoints and loss/gradient traces at the scheduled heartbeats remain private;scalar first/final traces
and training receipts are public. No checkpoint or feature data is committed.

All1,332 final expected training-record constraints remain positive. Mean final soft
intervention fraction is0.816040716 and maximum dual6.736239284. These are soft
training quantities,not final hard-policy risk or proof that longer training fixes it.
Total loss includes changing duals;do not compare its magnitude across steps as if
it were a fixed evaluation objective. No early checkpoint or seed was selected.
The separate frozen-budget policy evaluation determines actual utility and safety.

## Readout Directory Repair

The first readout PID78416 exits1 before completing a group because the checkpoint
writer expects allocation_progress/ to exist. The original log is retained.
Only this experiment's allocation_progress/ and progress/ directories are created;
evaluation resumes with the same registered script,features,weights,labels and2%rules.
No trained model or scientific registration is changed or rerun because of this I/O error.
Reproduction must create those directories before eval. This operational fix is
not evidence of learning improvement. Full portable first-run ergonomics remain
a separate packaging task;the current result is not advertised as one-command
anonymous from-scratch reproduction.
