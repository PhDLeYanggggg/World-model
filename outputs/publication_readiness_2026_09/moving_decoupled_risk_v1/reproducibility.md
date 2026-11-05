# Reproducibility

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_decoupled_risk.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_decoupled_risk.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_decoupled_risk.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_decoupled_risk.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_decoupled_risk.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_decoupled_risk.py
```

Completed phases refuse overwrite. Only resume an interrupted unchanged
phase with --resume. Exact replay uses existing checked artifacts, not
new data independence. This is a one-producer-seed model diagnostic.
