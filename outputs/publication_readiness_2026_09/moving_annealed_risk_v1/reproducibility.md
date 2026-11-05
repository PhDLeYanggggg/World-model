# Reproducibility

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_annealed_risk.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_annealed_risk.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_annealed_risk.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_annealed_risk.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_annealed_risk.py verify
```

On original completed outputs, phases refuse overwrite. Use unchanged --resume
only for interrupted phases. The pilot verifies exact resumed weights.
Reports can be regenerated with report_m3w_moving_annealed_risk.py.
Checkpoints and caches are private and must match the recorded hashes.
