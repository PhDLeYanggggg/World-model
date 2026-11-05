# Reproduction And Recovery

Native arm64 `.venv-pytorch/bin/python`; Torch CPU4/interop1, workers0 and four
numerical threads during sklearn fit/inference. No resource probing or new HPC
job. Atomic checkpoints every eight boosting rounds. Use `--resume` only for an
interrupted incomplete phase. Completed receipts and old results are immutable.

First execution in the registered namespace:

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_nonlinear_risk_readout.py tests/test_m3w_nonlinear_readout_exclusion.py tests/test_m3w_nonlinear_readout_reporting.py tests/test_m3w_frozen_readout_refit.py tests/test_m3w_readout_refit_reporting.py
.venv-pytorch/bin/python scripts/run_m3w_moving_nonlinear_readout.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_nonlinear_readout.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_nonlinear_readout.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_nonlinear_readout.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_nonlinear_readout.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_nonlinear_readout.py
```

After completion, rerun the report command to check all weights, parent hashes,
model pairing and frozen replay receipt without overwriting results. A fresh
full independent execution needs a new registered output namespace. Private
SOURCE arrays, parent weights and banks are not included in public Git. Public
light evidence alone does not reproduce raw inference. No raw data, cache,
images, weights or third-party datasets are committed. Focused tests do not
stand in for the complete historical test suite.
