# Execution and Recovery

Use the native arm64 environment from the repository root:

```sh
.venv-pytorch/bin/python scripts/audit_m3w_european_observation_quality.py --phase pilot
.venv-pytorch/bin/python scripts/audit_m3w_european_observation_quality.py --phase run
.venv-pytorch/bin/python scripts/report_m3w_european_observation_quality.py
.venv-pytorch/bin/python scripts/audit_m3w_european_observation_quality.py --phase verify
.venv-pytorch/bin/python scripts/verify_m3w_european_observation_quality.py
```

The committed registration precedes the pilot. `run` resumes completed
per-record receipts after checking their hashes and registration. `verify`
reparses every admitted raw member and requires exact diagnostic-array equality.
Private diagnostic arrays and logs remain under
`data/stage_cvpr2027_experiments/european_observation_quality_v1/` and are not
committed. `heartbeat.json` and `events.jsonl` record PID and record progress.
Check that PID before interpreting an old heartbeat as a running job. A
nonblocking process lock prevents concurrent writes; at least 10 GiB is reserved.
The code uses four compute threads, one interop thread and no worker processes.

CREATE was checked with the approved read-only queue helper. Its unrelated
pending jobs were neither submitted nor modified. Local execution is appropriate
for this sequential, bounded audit; no GPU training is claimed.

The first registration attempt failed before artifact creation because the
parent seal's artifact mapping was handled as a list. That reader was corrected
before registration and pilot execution. No observation result or metric was
discarded or selected by this correction.

The versioned neural adapter is `src/world_model/m3w_partial_context.py`.
It uses observed-token attention and past-only scaling over valid partial
observations. It does not modify the old factory or load an old checkpoint
into deployment. Matched training has not run; unit tests establish finite
gradients, parameter equivalence and token influence, not research improvement.

Replay checks cover the registered source cohort, input repair and reporting.
They do not establish prediction lift, retraining, a cold new-data download,
human annotation validity, independent confirmation or full legacy-suite success.
