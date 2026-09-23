# Past-Only External Predictor Interface

## Result and Scope

2026-09-23, `fresh_run` implementation and synthetic regression checks. The
adapter connects an explicit eight-step annotation prefix to the existing
476-dimensional predictor input. It does not open a dataset, admit a reserved
source, read future labels, or establish forecasting accuracy.

`ExternalPrefixAdapter` receives only raw `[frame, agent, x, y]` observations
at or before the query. It retains the last eight raw annotation frames, selects
prediction targets by complete past support, and retains incomplete-history
agents in the observed neighbor inventory. Missing future supervision cannot
remove an agent from the input. Neighbors without an immediately preceding raw
observation have no invented velocity/TTC estimate. All coordinates remain
dataset-local; one native annotation step is not a shared physical duration
across DUT, DroneCrowd and SDD.

The dense-array helper slices only those eight frames from position and validity
arrays. A regression fixture rejects whole-array conversion or reads outside the
prefix. Future-only agents are not observed, while invalid masked coordinates
are not promoted to valid points. There is no label-access method or learned
normalizer in this interface.

## Checks

- Complete prefixes reproduce the established numerical input exactly.
- Changing or removing future coordinates leaves inputs and observed agents unchanged.
- Incomplete past support is reported, not repaired with future data.
- Missing neighbor frames do not imply stationary motion or gap-spanning velocity.
- Duplicate keys, nonfinite observed positions and malformed dense arrays fail closed.
- A real Torch model accepts the synthetic input and produces finite `12 x 2` outputs.
- 21 new tests pass; 126 combined adapter, recording, reservation and admission tests pass.

```sh
.venv-pytorch/bin/python -m pytest \
  tests/test_m3w_external_prefix_adapter.py tests/test_m3w_causal_recordings.py \
  tests/test_m3w_dronecrowd_cache.py tests/test_m3w_dronecrowd_windows.py \
  tests/test_m3w_source_reservations.py tests/test_m3w_intake_admission.py -q
```

These are input-interface and regression results, not independent confirmation
or real-data model performance. No reserved raw arrays, forecasts or errors were
read to develop these fixtures. Source admission, label-side evaluation, the
complete frozen producer chain and the limits of independent calibration remain
separate requirements. Offline annotation prefixes do not certify sensor-time
causality of the original annotation process. Stage5C and SMC remain off.
