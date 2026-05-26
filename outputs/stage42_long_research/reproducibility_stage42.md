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

<!-- STAGE42_AC_REFRESH:START -->
## Stage42-AC Latest Evidence Refresh

- source: `fresh_synthesis_from_stage42_wxyz_aa_ab_artifacts`
- scope: protected dataset-local raw-frame 2.5D paper package only.
- Stage42-AB is now included as auxiliary-head evidence.
- Auxiliary-head conclusion: mixed / partial; not a uniform main contribution claim.
- Stage5C and SMC remain disabled.

### Additional Command

```bash
.venv-pytorch/bin/python run_stage42_full_waypoint_auxiliary_ablation.py
python3 -m pytest tests/test_stage42_full_waypoint_auxiliary_ablation.py tests/test_stage42_paper_package_refresh.py
python3 -m pytest tests
```
<!-- STAGE42_AC_REFRESH:END -->
