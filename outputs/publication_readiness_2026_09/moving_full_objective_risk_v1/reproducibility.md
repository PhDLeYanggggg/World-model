# Reproducibility

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_full_objective_risk.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_full_objective_risk.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_full_objective_risk.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_full_objective_risk.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_full_objective_risk.py verify
```

Use unchanged --resume only after an interrupted phase. Completed phases
refuse overwrite. Checkpoints and row caches remain private and must match
recorded hashes. The first complete fit is independently rerun exactly;
this is not an independent data test. Regenerate reports with
scripts/report_m3w_moving_full_objective_risk.py.
