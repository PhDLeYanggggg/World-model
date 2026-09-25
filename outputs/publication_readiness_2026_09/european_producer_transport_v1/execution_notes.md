# Execution and Provenance

## Completed Locally

Native arm64 Python3.11.1 / Torch2.12.0, CPU threads4, inter-op1, workers0,
batch128. No Conda/Rosetta runtime, DataLoader multiprocessing or resource probing.
All required phases terminated normally; no training process was stopped for
being slow. There were no fresh optimizer updates in this diagnostic.

| Phase | PID | Observed completion / result |
|---|---:|---|
| Source and registration checks |41810|2026-09-25 05:58:02 UTC, exit0 |
| 4,096-row inference pilot |41917|05:59:21 UTC, exit0;0.7553 seconds measured inference |
| All18 banks and72-view readout |41964|06:11:07 UTC, exit0 |
| Sampled inference plus full metric replay |42758|06:13:09 UTC, exit0 |

The eighteen banks contain3,827,628 predictor-row pairs and required582.2189
seconds of summed inference/cache-write time. This excludes setup, metric
calculation, second-pass replay, testing and report time. It is not3.8million
independent examples: the318,969-row source population is reused across models.
ADE/FDE support differs and is reported explicitly, including absent labels.

Nine existing full predictors, eighteen inner predictors and54 fitted score
heads are cached_verified, not fresh training. Each new bank uses only past
geometry. Each producer's first4,096 held-index predictions replay exactly;
this is not a random sample or full regeneration of all bank predictions. Full4 and damping
score arrays equal the original banks;72 original decision receipts reconstruct
exactly. Complete analysis reproduces. Raw ADE accounting independently matches
the frozen metrics, with raw FDE added to the registered report.

## Audit Anchors

Analysis SHA256:
`e6f6ea54e9bdca204c6660dc074dab9d2d9afaa2133557766e20c0d870dd835b`

Compact summary SHA256:
`369e15f1a3f7ce2757a34183d119a8ef81e47dcb6b80545360e850f6af893628`

Raw ADE/FDE metric SHA256:
`736d76030c46dd215e8bf7975e2f3366bb8eedfa8b4d2bb9cad0b28ae247b587`

The runner, configuration, inference helper, causal accounting and registration
stay frozen under the identity in matrix.json. Reporters add presentation and
registered raw-FDE accounting without changing predictions, scores or thresholds.
The previous nested analysis hash remains unchanged.

201 tests pass in30 scoped files; the complete legacy test suite was not run.
The test-file list and log hashes are in completion_checks.json. The saved
three-panel comparison was visually inspected: all36 comparisons and their
conditional intervals are present, including negative values.

## CREATE and Storage

The authorized read-only queue check succeeded at05:50:09UTC, observing2 running
and1 pending user tasks. It is a point-in-time queue observation, not evidence
of M3W training or completed remote assets. No remote tasks were submitted,
cancelled or modified. The M3W-specific remote asset directory is still unverified.
Local inference was appropriate given the measured runtime and available memory.

Only code, configuration, documentation and aggregate metrics belong in Git.
Raw data, prediction banks, source arrays, checkpoints and PNGs stay local and
ignored. The SVG is an aggregate scientific figure. Preserve unrelated staged
work; commit only the explicit experiment paths and the project result records.

No new calibration, reserved readout, deployment, Stage5C or SMC. Source-only
image pixels and raw steps do not support seconds, metric or physical-safety
claims. This completed experiment does not make M3W submission-ready.
