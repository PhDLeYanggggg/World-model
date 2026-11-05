# Reproducibility

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_signed_decision.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_signed_decision.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_signed_decision.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_signed_decision.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_signed_decision.py verify
```

Use --resume only for unfinished phases with unchanged registration.
Completed phases refuse overwrite. Checkpoints and prediction caches stay local.
No reserved data are needed. `report_m3w_moving_signed_decision.py` reproduces reports.
