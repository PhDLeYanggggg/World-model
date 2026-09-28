# Run and Recovery Guide

All commands use the native arm64 environment from the repository root.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/diagnose_m3w_fitting_switches.py register
# Commit the registration before reading fitting outcomes.
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/diagnose_m3w_fitting_switches.py pilot
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/diagnose_m3w_fitting_switches.py run --resume
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/diagnose_m3w_fitting_switches.py replay
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/report_m3w_fitting_switches.py
```

The completed receipts are immutable: do not overwrite them to manufacture a new run. The resume option verifies completed group inputs and continues missing groups after an interruption; a completed run is not restarted. A replay reconstructs every group and compares the results exactly without rewriting the group receipts.

Heartbeat and group receipts are under data/stage_cvpr2027_experiments/european_fitting_switch_diagnostic_v1/. Check the recorded PID and completed group count; slow progress is not a hang. CPU4, interop1 and workers0 are fixed. The runner stops before crossing the10GiB reserve plus32MiB temporary margin and preserves completed work.

Only code, configuration, aggregate reports and light receipts belong in Git. Do not commit private fitting data, checkpoints or unrelated staged files. Full legacy tests and independent confirmation are not run by this diagnostic.
