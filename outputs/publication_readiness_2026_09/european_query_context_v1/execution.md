# Runtime and Evidence Ledger

Registration/code commit `a3c8699d` was pushed before real fitting.

## Resource Check
Local native arm64 `.venv-pytorch`, Torch2.12.0, NumPy2.4.6, CPU4/inter-op1,
workers0. The real pilot ran300updates including direct/resumed replay, exited0
(PID58557),21.009s,2.554GB peak RSS. Interrupted50+50updates exactly reproduce
direct100updates, including optimizer, RNG, sampler and model state; only wall
time is excluded. Protected anchor outputs are exact. Initial objectives match.
The runtime estimate is391.82fit seconds with1.5margin, not an observed full run.
Checkpoint100updates; full scope remains36heads at2,000updates each.

Context is not a vacuous input in the pilot:81,088/81,958 rows of source008,
265/415 of020,1,470/1,470 of048,3,453/3,631 of067 have contemporaneous peers.
Maximum available query sizes are49,6,15,18 respectively. These are input-only
counts, not independent observations or outcome-based inclusion criteria.

CREATE read-only inventory succeeds and verifies24TRAIN packets. Its scheduler
query timed out after20seconds; this is unavailable queue observation, not a
failed training job or SSH access failure. No new job, authentication change,
restart, cancellation or remote write. Local training streams and SHA-verifies
those packets, retaining no new numerical data cache. Private inventory receipt:
`data/stage_cvpr2027_experiments/european_temporal_auxiliary_v1/query_context_readonly_20261029T024642Z.json`.

## Commands
Run from the project root, using the native environment:
```bash
.venv-pytorch/bin/python -m scripts.run_m3w_query_context_risk register
.venv-pytorch/bin/python -m scripts.run_m3w_query_context_risk pilot
.venv-pytorch/bin/python -m scripts.run_m3w_query_context_risk train
.venv-pytorch/bin/python -m scripts.run_m3w_query_context_risk train --resume
.venv-pytorch/bin/python -m scripts.run_m3w_query_context_risk eval
.venv-pytorch/bin/python -m scripts.run_m3w_query_context_risk check
```
`--resume` is only for an interrupted incomplete phase, never a completed phase.
Do not run train and train--resume together. The process lock rejects duplicates.
Freeze and commit all final training receipts before evaluation. Check performs
real inference, old-control replay, metric and bootstrap recalculation.

## Boundaries
Fresh real pilot; full training is running and scientific readout is pending.
No independent role opened, no deployment promotion, no new forecaster.
Detector-silver pixels, obs8/pred12 at rawstride12. No metric/seconds, physical
safety, human gold, true3D or foundation claim. Stage5C/SMC remain off.
