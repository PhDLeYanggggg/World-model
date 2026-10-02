# Positive Conditional Harm Control

Current status: full72 fresh fits completed, exact refits and inference replay
passed; registered scientific advance screen failed. This is cost-head training,
not a new neural trajectory decoder. No deployment upgrade.
Registered before real fitting at commit `c1556a2b`.

## Reproduction

Use the native arm64 environment from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_positive_harm.py register
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_positive_harm.py pilot
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_positive_harm.py run --resume
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_positive_harm.py verify
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.verify_m3w_positive_harm
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m scripts.report_m3w_positive_harm_slices
```

Do not rerun `pilot` after the full immutable result exists. `verify` performs
an expensive full refit/replay. An interruption resumes verified completed
groups; an incomplete group is refitted and any existing packet must hash-match.
No different threshold/model is chosen after validation readout.

Requires the registered local source assets, original frozen forest checkpoints,
past-quality receipts and authorized CREATE access. Old additive checkpoints
are read-only hash-verified packets; new weights are streamed to
`/users/k24101830/m3w/european_positive_harm_v1/inputs/`. They are not in Git.
Private heartbeat/logs are under
`data/stage_cvpr2027_experiments/european_positive_harm_v1/`.

Pilot:124.42s,6.80GB peak RSS,10.22MB checkpoint,184 independent scalar/anchor
checks; exact refit and serialized inference. First transport connection failed
before training; unchanged authorized connection retry succeeded. No access
restrictions were changed. The first full launch also failed before fitting on
the old-checkpoint read connection; unchanged retry succeeded. These were not
model failures or evidence of a repaired authentication system.

Full run:PID39862,1322.59s, peak10.087GB,72 fresh heads with exact refits,
152,397,838 checkpoint bytes, all owned-remote files rehashed.13,248 independent
scalar/parent checks and526 separate aggregate/bootstrap fields pass.37 scoped
tests pass; the full historical repository suite was not rerun for this change.
No local numeric cache or new scheduler job. Numerical training passed;
prediction and risk advance criteria did not. See [conclusions](conclusions.md),
[gates](gates.md), [failure analysis](failure_analysis.md), [method](method_note.md)
and [next action](next_action.md). `summary.json`, `slices.json`, `complete.json`,
`verification.json` and the72 group JSON files retain all controls and negative
results. Completed files are immutable and hash-bound; do not overwrite them.

Read [protocol.md](protocol.md) for the fixed loss and comparison. Twelve
already-exposed development localities; detector-silver obs8/pred12 rawstride12.
Independent selection, calibration and confirmation stay closed. No physical
safety, metric, seconds, human-gold, true3D, foundation or submission-ready
claim. Stage5C/SMC off. Historical SDD/UCY scores are not independent evidence.
