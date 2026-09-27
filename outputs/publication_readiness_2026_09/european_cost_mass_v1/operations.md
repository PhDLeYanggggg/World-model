# Run and Recovery

Use the repository root and native arm64 `.venv-pytorch/bin/python`. The runner
rejects Rosetta before Torch import, fixes CPU threads4/interop1, uses no worker
processes, writes PID/UTC heartbeats and obtains an exclusive experiment lock.
This run does not train Torch models or submit CREATE jobs. Resume is automatic
at per-view receipt boundaries; no `--resume` flag is required. Completed views
are verified before reuse. Partial unsealed outputs are recomputed. Preserve
10GiB free space and never delete unrelated data to make room.

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_cost_mass.py --phase register
# Commit registration_lock.json and its bound scientific sources before pilot.
.venv-pytorch/bin/python scripts/run_m3w_european_cost_mass.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_cost_mass.py --phase fit
# Commit prediction_freeze.json before evaluate.
.venv-pytorch/bin/python scripts/run_m3w_european_cost_mass.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_cost_mass_native.py
.venv-pytorch/bin/python scripts/plot_m3w_european_cost_mass.py
.venv-pytorch/bin/python scripts/diagnose_m3w_european_cost_mass.py
.venv-pytorch/bin/python scripts/verify_m3w_european_cost_mass.py
```

Private receipts, detailed fitting records, score arrays, test logs and heartbeat
are under `data/stage_cvpr2027_experiments/european_cost_mass_v1/`.
Detailed held costs live in the ignored public-directory `readout.json`.
Never commit either set. Public manifests contain hashes, not data.
The verification script repeats exactly the frozen readout; it does not select
another configuration. The source-held data are already exposed development
data. Neither this replay nor a good source interval constitutes independent
confirmation or permission to deploy.

The native report entrypoint repairs one diagnostic NumPy integer serialization
without changing the hash-bound scientific implementation. See
report_serialization_amendment.md for the exact boundary and regression tests.
