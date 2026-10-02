# Frozen Forest Decoder Experiment

This experiment isolates one feasibility-projection operation in the current
cost estimator. It completed on local arm64 CPU and is not a new training run.
See [conclusions](conclusions.md) and [failure analysis](failure_analysis.md).

## Reproduction

Required local assets are the hash-bound source forests, features and partitions
from the existing European development pipeline. No raw data or checkpoints are
included in Git. The protocol and source closure were committed before readout
in `a7049bd5`; no independent data role is opened.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m pytest tests/test_m3w_forest_projection.py tests/test_m3w_forest_projection_readout.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/verify_m3w_forest_projection.py
```

On a fresh result directory with the exact registered assets, run `register`,
`pilot`, then `run` phases of `scripts/run_m3w_forest_projection.py`.
Interrupted incomplete runs accept `run --resume`, reconstruct and reverify
existing groups, and continue without changing immutable outputs. Completed
runs intentionally refuse overwrite. Full inference and aggregates were already
computed twice for each of72 groups in this run; use the reader for cheap
post-completion artifact verification, not a false claim of retraining.

## Execution

- Pilot PID25445:14.39s, peak6.305GB; no scientific scope reduction.
- Complete PID25520, terminal exit0:53.5843s, peak9.796GB.
- Native arm64, Torch2.12.0, four compute threads, no DataLoader workers.
-72 cached-verified checkpoints; zero new model updates or CREATE jobs.
-6624 scalar/parent fields checked;3210 independently reduced aggregate fields.
-14 scoped tests passed; full historical suite not run.
-72 group JSON files total626,130bytes; no local numerical cache or checkpoint.
-Heartbeat and lock under the ignored experiment directory; resume supported.

Local free disk remained below the unchanged10GiB cache reserve. Computation
reused existing arrays in memory and wrote only small aggregate evidence.
No unrelated staged files, existing remote jobs or data were changed.

## Claims

This is source-only exploratory evidence on12 previously exposed localities.
It does not repair the deployment risk gate. No independent confirmation,
directional transfer, neural training or model promotion occurred.
Protocol remains obs8/pred12 raw stride12 and image-local detector-silver,
not metric, seconds, human-gold, physical safety, true3D or foundation evidence.
Stage5C and SMC remain off.
