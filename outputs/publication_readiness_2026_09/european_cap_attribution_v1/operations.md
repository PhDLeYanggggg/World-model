# Execution and Recovery

Use the native `.venv-pytorch/bin/python` from the repository root. Four
compute threads, one interop thread and zero loader workers. Existing Torch
models are reused; this experiment has no fitting or gradient updates.

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_cap_attribution.py --phase register
.venv-pytorch/bin/python scripts/run_m3w_european_cap_attribution.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_cap_attribution.py --phase freeze
# Commit prediction_freeze.json before evaluating the new contrasts.
.venv-pytorch/bin/python scripts/run_m3w_european_cap_attribution.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_cap_attribution.py
.venv-pytorch/bin/python scripts/plot_m3w_european_cap_attribution.py
.venv-pytorch/bin/python scripts/verify_m3w_european_cap_attribution.py
```

Registration commit efa546f3 precedes the new counterfactual outputs and their
scoring. The pilot verifies one view and all four recovered score vectors in
15.30s excluding ancestry verification; it adds no fitting and uses no new
scoring labels. Free disk at pilot completion: 12.52GiB. The runner stops on
less than 10GiB free; it does not delete unrelated assets or downgrade scope.

Each inference receipt binds parent models, feature matrices, row alignment,
raw predictions, envelope and recovered scores. Receipt-checked continuation
keeps completed work unchanged. Scores are recoverable from frozen models;
hash receipts avoid another large data cache. An exclusive process lock avoids
duplicate execution. Heartbeat.json and events.jsonl retain PID, phase, view
count and timestamp. A live progressing process is not restarted for slowness.

The final verifier is single-use: replay all recovered vectors and scoring,
check byte-identical aggregate reports and figure, run focused regressions,
and seal source and public-artifact hashes. For later repeat checks use the
runner's verify phases and new log paths, not overwritten attested logs.
This verifies existing assets, not a cold rebuild from downloaded raw data.

CREATE was queried read-only at 2026-09-27T06:44:39Z. Its three unrelated jobs
remained pending for priority/dependencies; none was submitted or modified.
This does not imply those jobs failed. This experiment fits local resources.

Private per-view receipts, detailed metrics and logs are excluded from Git.
Commit only owned code, configs, public aggregate reports and root result
records. Existing unrelated staged work must be preserved.
