# Reproduction

Registration commit a0d63f52 preceded new decisions. Parent199 public artifacts,
source bindings,144 trained neural heads,72 ridge fits and parent frozen decisions
are hash-verified. No new model fitting. No reserved roles opened.

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_incremental_joint.py --phase decide --resume
.venv-pytorch/bin/python scripts/run_m3w_european_incremental_joint.py --phase evaluate --resume
.venv-pytorch/bin/python scripts/report_m3w_european_incremental_joint.py
```

Use the reporter on completed artifacts; resume retains immutable group receipts.
Only one process can own the run lock. Heartbeat/events and per-group decisions
are private and resumable. Native arm64 CPU4/inter-op1, no loader multiprocessing.
The authorized CREATE queue observation was read-only; this small inference task
required no remote job or upload. No remote artifact directory was assumed.

Public files are aggregate reports/config/code, not raw tracks, forecasts,
per-agent outcomes or checkpoints. Reproduction needs the verified local asset
chain. Full legacy suite is not_run; scoped checks are recorded separately.
