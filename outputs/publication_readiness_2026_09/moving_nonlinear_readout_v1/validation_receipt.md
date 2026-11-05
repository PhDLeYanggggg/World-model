# Verification Receipt

- Real-data pilot: observed exit 0; both resumed/uninterrupted arms exact.
- Six primary fits: observed exit 0; 1,536 scalar boosting rounds and an
  additional complete independent first-fit repeat. No new encoder updates.
- Evaluation retry PID 73703: observed exit 0, 61.44913195799745 seconds.
- Independent replay PID 73820: observed exit 0, 61.9484606249971 seconds.
  All three inference/choice/metric blocks and all original controls exact.
- Reporting CLI: observed exit 0. A subsequent audit verified all six actual
  models, their frozen parents, paired matrices and exact replay receipt.
  All 34 then-existing public artifacts were byte-identical after rerunning
  the report. This receipt is added after that comparison.
- Research-state numerical metrics exactly match analysis and fit receipts.
- Focused tests: 15 passed in 4.06 seconds, observed exit 0. Full historical
  suite not rerun; this is not a claim that every project test passed.

Test command:

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_nonlinear_risk_readout.py tests/test_m3w_nonlinear_readout_exclusion.py tests/test_m3w_nonlinear_readout_reporting.py tests/test_m3w_frozen_readout_refit.py tests/test_m3w_readout_refit_reporting.py
```

Preserved engineering failures: the original synthetic fixture omitted parent
metadata and was fixed before real fitting; the first evaluation failed on a
nested preprocessing lookup and required the explicit exclusion-interface
amendment. Neither changed the scientific rules or fitted weights. A later
report-audit wrapper using a relative `runpy` path failed before writing any
artifact. Retrying that wrapper with the absolute script path succeeded;
the normal reporting CLI and all scientific results were unchanged.

Engineering reproduction is successful. Scientific expansion, original risk
and deployment gates fail. The goal remains active and submission readiness
is false. No identity certification, new H, Stage5C execution or SMC.
