# Local Operation and Reproduction

Use native arm64 `.venv-pytorch/bin/python` from the repository root with
`PYTHONDONTWRITEBYTECODE=1`. Runner:
`scripts/run_m3w_component_calibration.py`.

1. `register`, then commit config, protocol, code, tests and registration.
2. `pilot`: fit the first real source-recording margin, not synthetic-only QA.
3. `fit --resume`: verify the pilot and complete72 frozen-head calibrators.
4. Commit `calibration_freeze.json` before transferred action generation.
5. `replay_fit`: recompute all72 source calibrators exactly.
6. `decide`: freeze216 causal head/direction action views; commit the freeze.
7. `evaluate`, then `replay_evaluate` for complete outcome-accounting replay.
8. `report`: aggregate all fixed arms, bootstrap and scoped tests; seal outputs.

Source calibration records are under
`data/stage_cvpr2027_experiments/european_component_calibration_v1/fits/`.
They contain coefficients and recording-level aggregates, not new trajectory
caches or model weights. The public manifest binds their hashes. Original72
tree checkpoints and upstream assets stay private and unchanged.

A process lock prevents duplicate runs. PID heartbeat/events are saved after
every source-head group and during transfer readout. Interrupted fitting resumes
with `fit --resume`; identity/input hashes must match. Existing immutable results
must not be overwritten after changing code or scientific settings. A changed
design requires a separately registered version.

The pilot took12.9834s at6,090,293,248-byte peak RSS. Projected additional storage
is24,786,800 bytes, preserving the10GiB reserve. CPU4/interOp1/workers0.
The workload is feasible locally; no new CREATE job is needed. Completed job
37602475 belongs to an earlier verified replication, not this experiment.

These are empirical source-recording calibration parameters, not a new neural
world model. Independent confirmation remains closed.12 previously exposed
localities, obs8/pred12 stride12 raw frames, image-local detector-silver only.
No metric, seconds, human-gold, true3D, foundation or physical-safety claim.
Stage5C execution and SMC remain disabled. Only explicitly recorded scoped
tests count as run; the full legacy suite is not claimed.
