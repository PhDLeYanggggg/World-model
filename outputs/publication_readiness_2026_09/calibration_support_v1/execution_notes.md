# Reproduction

```bash
.venv-pytorch/bin/python -m pytest tests/test_m3w_calibration_support.py tests/test_m3w_experiment_contract.py tests/test_m3w_risk_calibration.py -q
.venv-pytorch/bin/python scripts/audit_m3w_calibration_support.py
.venv-pytorch/bin/python scripts/audit_m3w_calibration_support.py --verify
```

Tests session25748: 63 passed in 17.91s, exit0. Audit94315 and verify62432 exited0.
The script streams hashes of bound artifacts but deserializes only JSON metadata,
not checkpoint weights or trajectory arrays. Parent readout/replay receipts and
head identities are checked. Changed overwrites are refused; verification
requires exact recorded equality.

Fresh DUT screen, preserving prior evidence:

```bash
.venv-pytorch/bin/python scripts/build_m3w_dut_admission_screen.py --output outputs/publication_readiness_2026_09/calibration_support_v1/dut_screen.json --report outputs/publication_readiness_2026_09/calibration_support_v1/dut_refusals.json
```

This exited0 with 112 expected refusals and zero admissions. Do not rerun into
existing paths; that command deliberately requires new outputs. It validates
an old cache, not a fresh conversion or clearance of historical exposure.

All task executions are terminal. No new model training or calibration occurred.
The nonhermetic legacy full suite was not rerun. The 3019 unrelated staged
entries retain fingerprint
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.
Only scoped code, tests, reports, README and research state are committed.
