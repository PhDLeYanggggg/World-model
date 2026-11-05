# Reproduce And Verify

Use native arm64 `.venv-pytorch/bin/python`, CPU4/interop1/workers0. The runtime
records PID, heartbeat, peak memory and immutable phase receipts. All private
dependencies, checkpoints and cache hashes must match the registered parents.

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_frozen_readout_refit.py tests/test_m3w_readout_refit_reporting.py tests/test_m3w_direct_tail_risk.py tests/test_m3w_direct_tail_reporting.py
.venv-pytorch/bin/python scripts/run_m3w_moving_readout_refit.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_readout_refit.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_readout_refit.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_readout_refit.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_readout_refit.py
```

Completed experiment phases intentionally reject overwriting receipts. Use
`--resume` only on an interrupted incomplete phase; completed fits are reused
after binding checks. The report command re-audits checkpoints and hashes
without retraining or using a new holdout. A new independent raw-data rerun
requires a separately registered namespace, not deleting frozen artifacts.
Four primary solves include three fits and one reproduction, not four seeds.
Private licensed data and pretrained parents are required: public Git alone
does not reconstruct them. No raw data, feature/cache arrays, weights or third-
party material is committed. Focused tests are not the full historical suite.
