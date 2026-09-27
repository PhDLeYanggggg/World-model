# Reproduction

Use the existing native arm64 .venv-pytorch; CPU 4/interop 1, workers 0. No resource
probing, multiprocessing or NumPy replacement for training. Private data and
checkpoint caches are not distributed. Registration binds source and parent
seals; original fitting states, forecasts and split lineage must be available.

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_intervention.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_intervention.py --phase train --resume
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_intervention.py --phase decide --resume
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_intervention.py --phase replay_heads --resume
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_intervention.py --phase replay_decisions --resume
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_intervention.py --phase evaluate
.venv-pytorch/bin/python scripts/run_m3w_european_dimensionless_intervention.py --phase verify_eval
.venv-pytorch/bin/python scripts/report_m3w_european_dimensionless_intervention.py
```

Register before fresh training; commit decision_freeze.json before readout.
On completed artifacts use replay phases; do not delete checkpoints to restart.
The first pilot resumes inside its 2,000-update budget. Checkpoint every 200 updates,
heartbeat and PIDs in the private event log. At least 10 GiB free before each fit.
Resume skips hash-verified completed heads and completed decision groups.
Shared simulation jobs/environment must never be modified. Local fit is measured
to fit the machine. CREATE queue readout is not M3W remote training evidence.

Report complete source replay separately from independent confirmation, raw-data
rebuild, formal calibration, deployment and full historical test-suite coverage.
None of the latter follow from a successful cache replay.
