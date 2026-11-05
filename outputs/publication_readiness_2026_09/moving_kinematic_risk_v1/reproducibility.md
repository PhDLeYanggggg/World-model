# Reproduction

Use native arm64 .venv-pytorch, CPU4/interop1/workers0; no resource probing.
Registered definitions and receipts are immutable. The first execution is:

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_kinematic_risk.py tests/test_m3w_direct_tail_risk.py tests/test_m3w_kinematic_feature_provenance.py
.venv-pytorch/bin/python scripts/run_m3w_moving_kinematic_risk.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_kinematic_risk.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_kinematic_risk.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_kinematic_risk.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_kinematic_risk.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_kinematic_risk.py
```

Interrupted incomplete phases use --resume with the same registered settings.
Completed receipts are not overwritten. The report can rerun to audit actual
weights, lineage, full fitting losses and exact saved readout. A new independent
execution needs a separate registered namespace. Private SOURCE annotations,
forecast banks and checkpoints are not in public Git; light reports alone
cannot reproduce raw inference. Focused checks do not replace the full suite.
