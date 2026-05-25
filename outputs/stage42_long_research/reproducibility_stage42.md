# Stage42 Reproducibility

## Environment

- training/eval scripts: `.venv-pytorch/bin/python` arm64
- tests: `python3 -m pytest tests`
- num_workers: `0` for torch data paths
- Stage5C executed: `False`
- SMC enabled: `False`

## Commands

```bash
.venv-pytorch/bin/python run_stage42_data_calibration.py
.venv-pytorch/bin/python run_stage42_external_validation.py
.venv-pytorch/bin/python run_stage42_full_waypoint_dynamics.py
.venv-pytorch/bin/python run_stage42_causal_ablation.py
.venv-pytorch/bin/python run_stage42_safety_floor.py
.venv-pytorch/bin/python run_stage42_paper_package.py
python3 -m pytest tests
```

## Source Labels

All Stage42 package claims use `fresh_run`, `cached_verified`, or `not_run`. Stage42-D explicitly does not relabel cached component ablations as fresh retraining.
