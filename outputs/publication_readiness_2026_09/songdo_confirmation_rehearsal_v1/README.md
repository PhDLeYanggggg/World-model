# Songdo confirmation runner

The real forecast packets have 200 flights, 600 seed views and 573
matched seed views across 17 linked source components. The runner
processed that roster with fabricated future displacements, not
Songdo outcomes. The receipt records counts, input hashes and no
effect sizes.

The matched comparison always keeps the two-edit budget and common
radius fixed. A separate readout reports what a release policy would
actually output, including CV fallback. A no-edit flight does not
count as a 2% risk pass. The original all, easy and signed easy checks
are unchanged. Missing future steps remain masked. Intervals average
three seeds within flight and then flights within each linked source
component before 3,000 component bootstrap draws.

## Reproduce without opening outcomes

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/rehearse_m3w_songdo_confirmation.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_songdo_confirmation.py tests/test_m3w_songdo_matched_readout.py tests/test_m3w_songdo_future_join.py
```

The small tests cover linked-site rejection, tied scores at the
calibration boundary, interruption and resume, immutable thresholds
after a confirmation failure, and the zero-shot path. A synthetic ZIP
also checks that coordinates are parsed only for indexed rows and
joined by actor, drone, visibility and timestamp.

## Real run, after protocol approval

No production manifest or real future result exists for this run.
The approved manifest must identify the approval, every site's role,
the native-unit easy/hard cutoffs and the release rule. It cannot split
linked sites. In calibrated mode, the tie-complete passing-score prefix
is fitted separately for each arm and seed using only calibration
flights. In zero-shot mode, every threshold and its source must be
declared in the manifest before any outcomes are opened.

```sh
.venv-pytorch/bin/python scripts/run_m3w_songdo_confirmation.py register --protocol /path/to/approved.json
# Calibrated mode only:
.venv-pytorch/bin/python scripts/run_m3w_songdo_confirmation.py calibrate --protocol /path/to/approved.json
.venv-pytorch/bin/python scripts/run_m3w_songdo_confirmation.py confirm --protocol /path/to/approved.json
```

Registration binds the protocol, inputs, source grouping and scorer
code. The local run directory stores the release freeze, a PID
heartbeat, one checkpoint per completed flight and the final result.
Repeat the same command after an interruption. A completed run returns
the saved result without reopening outcomes. Report errors in native
coordinates on the 8-observed/12-predicted-step grid. No metric,
deployment or cross-domain success claim follows from this rehearsal.
