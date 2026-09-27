# Reproduction and Recovery

This contrast requires the existing sealed source geometry and grouped control
bank. It is not a cold rebuild from downloaded raw data. The runner validates
parent seals, source bindings, control checkpoints and predictions; it fails
closed on mismatch. Do not overwrite a sealed experiment to make a rerun pass.

Use the native arm64 `.venv-pytorch/bin/python` from the repository root.
CPU compute threads4, interop1 and DataLoader workers0 are separate settings.
The architecture guard runs before Torch import. No hardware resource probe is
needed. A real pilot, not successful import, checks training and memory.

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_refit.py --phase register
# Commit registration before fitting. Do not rewrite its files after this point.
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_refit.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_refit.py --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_refit.py --phase predict
# Commit the complete prediction_freeze.json before comparative scoring.
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_refit.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_dimensionless_refit.py
.venv-pytorch/bin/python scripts/probe_m3w_dimensionless_units.py
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_refit.py --phase replay
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_refit.py --phase verify_eval
```

The pilot resumes into the first 4,000-update endpoint. Atomic checkpoints
include optimizer/schedule/RNG/sampler state; identity matching prevents an
accidental incompatible resume. A process lock prevents concurrent writers.
Completed endpoints are reused only after identity/hash verification. If the
process stops, inspect the logged PID and terminal status before resuming.
Do not restart because a log read or scheduler query times out.

Private run root:
`data/stage_cvpr2027_experiments/european_dimensionless_refit_v1/`.
Each trial has its checkpoint and completion receipt; `events.jsonl` and
`heartbeat.json` record PID, state, steps, loss, gradients and timing. Log
complete process wall time and maximum RSS separately from cumulative fit time.
Every200updates checkpoint; every50updates heartbeat. Preserve10GiB free disk.

Hashes and exact forecast/scoring replay verify this version, not independent
generalization. Report scoped test coverage separately from the unrun full
historical suite. CREATE access is read-only for this local-sized experiment;
existing remote jobs are not evidence that M3W itself trained remotely.

Before comparative outcomes are scored, an additional verification-only replay
is fixed to the first endpoint (fold0, seed17), not chosen by performance:

```sh
.venv-pytorch/bin/python scripts/replay_m3w_dimensionless_training.py
```

It trains from initialization for another4,000 updates in a separate private
directory. All weights, optimizer state, logged losses and sampler/RNG states
must match the frozen original. It is computational replication, not an extra
candidate, independent seed or cold raw-data rebuild. Account for its compute
separately from the nine-model experimental budget. The script refuses to
overwrite an existing replay checkpoint.
