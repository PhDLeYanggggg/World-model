# Execution and Recovery

This is a registered source-development mechanism experiment, not a model
promotion. Independent selection, reserved calibration and confirmation stay
closed. Native arm64 Python, CPU4, interop1 and a single process are used.
No MPS probing or DataLoader multiprocessing is needed. No remote job is
submitted; the fresh CREATE inspection was read-only.

## Ordered Commands

Run from the repository root with `.venv-pytorch/bin/python`:

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_regime_transport.py --phase support
# Commit registration and support before proceeding.
.venv-pytorch/bin/python scripts/run_m3w_european_regime_transport.py --phase pilot --resume
.venv-pytorch/bin/python scripts/run_m3w_european_regime_transport.py --phase train --resume
# Commit prediction_freeze.json before any held scoring.
.venv-pytorch/bin/python scripts/run_m3w_european_regime_transport.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_regime_transport.py
.venv-pytorch/bin/python scripts/plot_m3w_european_regime_transport.py
.venv-pytorch/bin/python scripts/verify_m3w_european_regime_transport.py
```

Registration binds the scientific code, configuration, tests and protocol.
Do not silently edit them after registration. The support and prediction
freeze must be committed before dependent phases, which enforce that order.
Figures and the final verifier do not change fitting or evaluation rules.

## Checkpoints and Monitoring

Private directory:
`data/stage_cvpr2027_experiments/european_regime_transport_v1/`.
`heartbeat.json` contains PID, UTC time and the latest state;
`events.jsonl` retains the event sequence. Training saves optimizer, model,
sampler state, draw counters and fixed diagnostic batch every200 updates.
It uses an exclusive lock. Never start a competing run or delete a lock
holder's checkpoint. A slow active process is not a failure.

Completed checkpoints are losslessly gzip-compressed and hash-receipted.
Resume checks identity and recovers a completed archive lacking only its
receipt under the exclusive lock. Partial raw checkpoints resume normally;
each head retains the fixed2000-update total. Preserve at least10GiB free.
Large arrays, models, row-level metrics and logs remain local and ignored.
Only code, configuration, reports and light aggregate metrics are committed.

## Reproduction Boundaries

The real-data pilot must reproduce every native three-site parameter exactly
before the crossed heads can train. Final verification replays predictions
and held metrics from retained checkpoints; it does not retrain all heads
from scratch. The native bridge is a genuine fresh retraining control.
Byte-matched reports and scoped tests establish reproducibility, not a
passed scientific gate. The full legacy suite is not silently claimed.

Obs8/pred12 native steps, detector pixels; no metric/seconds, human-gold,
true3D, foundation or physical-safety claim. Stage5C and SMC remain disabled.
