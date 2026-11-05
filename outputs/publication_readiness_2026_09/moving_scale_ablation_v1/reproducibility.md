# Reproducibility

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_scale_ablation.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_scale_ablation.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_scale_ablation.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_scale_ablation.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_scale_ablation.py verify
```

Resume only unfinished phases with --resume and unchanged registration_v2.json.
Completed phases refuse overwrite. Checkpoints/caches stay local and must match hashes.
report_m3w_moving_scale_ablation.py reproduces fitting diagnostics and reports.
