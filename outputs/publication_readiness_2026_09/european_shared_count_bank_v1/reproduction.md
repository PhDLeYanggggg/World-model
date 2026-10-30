# Reproducing the common-count control

## Local requirements

Run from the repository root using native arm64 `.venv-pytorch/bin/python`,
not Intel Conda under Rosetta. Keep CPU threads4, interop1, DataLoader workers0.
The runner checks architecture before importing Torch. It uses the existing
local verified source packets, frozen trajectories and18 cost checkpoints.
Those third-party data, private row IDs and weights are deliberately not in Git.
This is not a verified data-free cold-start reproduction claim.

The published aggregate archive can be verified without raw data. Regenerating
the current readout report also needs the registered code/config/dependency
receipts; runtime receipts are public. A new experiment must use a new identity,
not overwrite this registration or a completed pilot/readout.

## Original complete workflow

```sh
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_shared_count_bank.py tests/test_m3w_shared_count_readout.py
.venv-pytorch/bin/python scripts/run_m3w_shared_count_bank.py register
# Commit scientific code, config, protocol and registration before execution.
.venv-pytorch/bin/python scripts/run_m3w_shared_count_bank.py pilot
.venv-pytorch/bin/python scripts/run_m3w_shared_count_bank.py run
.venv-pytorch/bin/python scripts/run_m3w_shared_count_bank.py verify
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_shared_count_bank_v1
.venv-pytorch/bin/python scripts/report_m3w_shared_count_bank.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_shared_count_artifacts.py
```

Register and pilot refuse incompatible or already completed artifacts. `run`
does not overwrite a completed result. On a genuinely interrupted unfinished
run, inspect its heartbeat/PID/log first and use:

```sh
.venv-pytorch/bin/python scripts/run_m3w_shared_count_bank.py run --resume
```

Do not start a duplicate healthy process. Private progress binds inputs, model,
action availability, query IDs and settings; changing any of them is not resume.
Heartbeat events live under the same experiment identity in
`data/stage_cvpr2027_experiments/`; finished runtime and replay receipts record
the actual costs. A healthy run may continue up to12h with the fixed roster.

## Audit the published aggregates

```sh
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_shared_count_bank_v1 --verify
.venv-pytorch/bin/python scripts/report_m3w_shared_count_bank.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_shared_count_artifacts.py
```

The archive contains aggregate metrics, not row data or models. Its manifest
checks exact archived bytes against the completed roster. Summary reduction
uses complete-roster order, not archive lexical order. A successful report
rebuild or exact replay is not independent scientific confirmation.

Do not rerun this experiment to select a better outcome. Read all72 groups,
the predeclared primary pair, unknown sensitivity and original recording-risk
failure together. No metric/seconds-level, human-gold, true3D or foundation
upgrade; Stage5C and SMC stay disabled.
