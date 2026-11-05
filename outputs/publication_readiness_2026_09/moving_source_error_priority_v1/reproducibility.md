# Reproducibility

```sh
.venv-pytorch/bin/python scripts/run_m3w_moving_source_error_priority.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_source_error_priority.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_source_error_priority.py teachers
.venv-pytorch/bin/python scripts/run_m3w_moving_source_error_priority.py students
.venv-pytorch/bin/python scripts/run_m3w_moving_source_error_priority.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_source_error_priority.py verify
```

Use --resume only for an interrupted unfinished phase with identical registered files.
Completed phases intentionally reject overwriting. Reuse verified artifacts or a
separately registered rerun location. Both full refits are already exact.
Checkpoints, data, private prediction caches and local environment are not in Git.
The operative registration is registration_v2.json. No reserved data are needed.
