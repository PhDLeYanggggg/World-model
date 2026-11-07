# DroneCrowd Input Compatibility

The existing DroneCrowd offline XML cache now connects to the unchanged native
8-observation/12-prediction tensor packer. This is an interface check, not an
admission or new predictive result. I used three past prefixes from each of the
112 recordings, selected by clip boundaries rather than future availability.

The [later causal stream check](../m3w_causal_forecast_stream_v1/README.md)
preserves these exact prefixes and replaces the rough full-schedule estimate
with complete past-only packing and a real Torch recovery probe on development data.

- All 336 input queries pass. They contain 48,362 currently visible agent
  instances and 48,183 with the eight contiguous past steps required by the
  existing model. All currently visible agents remain in the upstream inventory.
  The frozen packer uses only eligible targets and its original eight-neighbor
  history rule; no claim that every visible agent enters the model.
- The adapter uses backward finite differences for the CV input and rejects
  future fields. A synthetic future-position/visibility perturbation leaves
  the actual packed features unchanged. Eleven scoped adapter/source tests pass.
- The recording cache integrity check reads hashes of the complete stored arrays;
  only past-prefix slices are supplied to the packer. No future label API, external
  forecast inference, loss readout, fitting, role assignment or deployment runs.

## Compute Implications

The full existing schedule contains 31,472 query times. Prior structural
counts imply roughly 4.52 million past-supported target instances across all
112 recordings, not independent examples. The previous native arm64 CPU4
dual-view inference measured 943.84 seconds for 251,080 rows across two arms.
At that throughput, one complete model pass over this external roster would
take about 4.7 hours before scoring, a projection rather than a new timing.
Three full passes could exceed 14 hours. Roles and the exact evaluation set
remain pending; this is not permission to sample fewer rows after readout.

Use the mmap reader and stream by recording. A dense 476-float materialized
feature table would require roughly 8.6 GB, before labels or forecast artifacts.
On the current disk, do not materialize it. Reuse local assets for development;
use CREATE for the full-scale phase if it is available and its data placement
and storage are verified, without starting duplicate jobs.

This remains an offline image-coordinate task at raw annotation stride 1.
Annotation sensor-time provenance, seconds and metric scale remain unverified.
No new policy certificate, independent result, Stage5C, SMC or submission claim.

```sh
.venv-pytorch/bin/python scripts/check_m3w_dronecrowd_frozen_inputs.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_dronecrowd_frozen_inputs.py tests/test_m3w_dronecrowd_cross_archive.py tests/test_m3w_dronecrowd_grouping.py
```
