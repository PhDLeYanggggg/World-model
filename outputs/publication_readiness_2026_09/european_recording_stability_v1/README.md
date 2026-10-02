# Recording-Deletion Stability Control

Status: registered implementation; real nested calibration not_run yet.
The parent experiment is complete and negative. This is a new source-only
diagnostic, not a transfer run, a promoted model or independent confirmation.

The [protocol](protocol.md) fixes both calibration arms and the entire72-head
roster. Parent leave-one-recording-out actions must replay before any deletion
diagnostic. Held-recording outcomes are excluded from every nested calibrator.
Zero support means unestimable, not safe.

## Run and Resume

From the repository root, using the existing native arm64 environment:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m pytest -p no:cacheprovider tests/test_m3w_recording_stability.py -q
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_recording_stability.py pilot
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_recording_stability.py run
```

If an execution is verified interrupted, use the last command with `--resume`.
Existing aggregates are reproduced exactly, never silently replaced. A completed
run must not be launched again. Logs expose the PID, current group, elapsed time
and peak memory. The private heartbeat is at
`data/stage_cvpr2027_experiments/european_recording_stability_v1/heartbeat.json`.

All source inputs are streamed from the already completed CREATE experiment.
No new HPC job, login-node fitting, local raw/array cache, or simulation changes.
Small group summaries are the checkpoints; all nested computation has exact
replay. Frozen code/config/protocol hashes are in `registration.json`.

The same-count comparator is an exact expected utility under uniform thinning
within a recording. It is not same-frame random selection, a realized policy,
an expected ratio, or a safety certificate. Independent roles remain closed,
the original2%risk budget remains, and Stage5C/SMC stay off.
