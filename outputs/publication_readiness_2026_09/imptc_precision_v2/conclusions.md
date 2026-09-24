# Canonical Precision Repair: Engineering Check Passed

The fixed 1e-9 dimensionless prefix precision resolves the previously observed
three EqMotion unit-probe failures on the same registered 58 queries / 754 windows.
Both Transformer and EqMotion geometry, normalized predictions and cost features
are exactly equal across .01x/1x/100x coordinate representations. No moving-to-static
or static-to-moving changes were observed in these 754 windows. Four new regression
tests pass, including an explicit example where sub-precision motion *can* vanish.
The complete numerical-array and report replay is exact.

This is a fresh numerical probe on cached-verified source prefixes and frozen
weights. It is not new predictor training, a forecast-error evaluation, a lossless
representation guarantee or predictive lift. The old failed evidence remains
unchanged. The precision was fixed before this run and no future endpoint/errors
were consulted. There is no general claim covering arbitrary coordinate values,
sensor quantization or unseen recordings. Tiny motion may be lost in other data.

The actual source remains a single-intersection sample with mixed road-user
types and unverified upstream online-processing causality. Its covariates are
already engineering-design-exposed, not untouched confirmation. A native
master-index 8/12 request does not establish matched SDD time or metric accuracy.

This new prefix version is deliberately **not** inserted into the concurrent
dimensionless-risk training matrix: that experiment fixes old source geometry
and forecasts to isolate head features. Combining both changes now would break
the registered matched comparison. No old checkpoint, policy or data role is
silently replaced. Deployment, DroneCrowd confirmation, Stage5C and SMC remain
unchanged/closed.

## Reproduce

```bash
.venv-pytorch/bin/python scripts/audit_m3w_imptc_precision.py --verify
.venv-pytorch/bin/python -m pytest tests/test_m3w_quantized_prefix.py -q
```

The probe uses native arm64, CPU2, interop1, no multiprocessing. Private numerical
arrays are ignored by Git; only the source, registration and lightweight results
are published. A successful input test is not independent research confirmation.
