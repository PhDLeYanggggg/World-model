# Operation and Resume

Use native `.venv-pytorch/bin/python` from the repository root. Four compute
threads, one interop thread, zero loader workers. Cached Torch neural heads are
restored for genuine inference; new models are explicitly closed-form ridge
probes, not a Torch-training or NumPy-fallback success claim.

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_temporal_support.py --phase register
.venv-pytorch/bin/python scripts/run_m3w_european_temporal_support.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_temporal_support.py --phase fit
# Commit prediction_freeze.json before the following evaluation.
.venv-pytorch/bin/python scripts/run_m3w_european_temporal_support.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_temporal_support.py
.venv-pytorch/bin/python scripts/plot_m3w_european_temporal_support.py
.venv-pytorch/bin/python scripts/diagnose_m3w_european_temporal_support.py
.venv-pytorch/bin/python scripts/verify_m3w_european_temporal_support.py
```

The fitting phase automatically verifies completed per-view receipts and
continues missing fits. It does not reset old models. Private heartbeat.json
and events.jsonl record PID, phase, view count and UTC. An exclusive lock avoids
duplicate runs. The runner reserves at least 10GiB free disk.

The final verifier is a single-use attestation. Subsequent repeat checks use
`--phase verify_fit` / `--phase verify_eval` directly and new log paths, never
overwrite attested logs. All completed unchanged checks can be reused after
hash validation. Exact replay is against current cached assets, not a cold
raw-data rebuild or independent result.

Pilot: four fitting-only probes completed in 15.48s excluding ancestry checks,
12.75GiB free at completion, no scoring labels read. CREATE was checked read-only
at 2026-09-27T05:59Z; unrelated pending jobs were preserved. No M3W HPC job was
submitted. This compact information screen fits local compute and disk limits.

Public artifacts contain code, configuration, protocol and aggregate evidence.
Detailed row predictions, model coefficients and source/lineage caches remain
under the Git-ignored private data directory. Do not stage unrelated changes.
