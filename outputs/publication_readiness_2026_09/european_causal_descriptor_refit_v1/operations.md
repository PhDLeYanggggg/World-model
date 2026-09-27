# Run and Reproduction Record

## Execution

Native arm64 `.venv-pytorch/bin/python`, CPU threads 4, interop threads 1,
workers 0. No torch device/resource probing or NumPy training replacement.
Atomic lossless gzip checkpoints preserve optimizer/RNG/draws and resume
after the pilot. The 100 pilot updates are included in each 2,000-update
training budget. Replays are verification compute, not additional selected
training trials.

| Operation | PID | Process wall seconds | Maximum RSS bytes | Outcome |
|---|---:|---:|---:|---|
| 100-update pilot | 34510 | 7.24 | 6091177984 | Resumed into fixed budget |
| Complete 108-head training | 34588 | 314.49 | 11125374976 | 216,000 updates |
| Frozen readout | 35127 | 129.79 | 9854025728 | Primary and risk gates failed |
| First complete fit replay | 35210 | 10.36 | 4707418112 | Model/optimizer/RNG/draws/losses exact |
| Full prediction/decision replay | 35374 | 120.63 | 10026369024 | 108 groups exact |
| Full evaluation replay | 35600 | 129.60 | 10168336384 | Summary/details exact |

The first real inference replay also removes future fields and verifies exact
outputs plus equal initial functions. This is one explicit removal replay,
not a claim that all 108 training jobs were refitted from initialization.

The final verifier checks all matched control chains, independently reconstructs
same-current-query counts and independently reduces locality metrics. Exact
counts and the scoped test log are in `verification.json`. The full legacy
test suite and a cold raw-data rebuild remain **not_run**.

## Storage and Compute Choice

The private study occupies approximately 59.1 MiB after replays. Checkpoints
are compressed losslessly; actions use compressed boolean arrays. Score
hashes are frozen and scores are recomputed, avoiding another duplicated
large score bank. A post-readout free-space check reports about 10.56 GiB;
preserve the 10 GiB reserve and recheck before the next study. No unrelated
data was removed.

The approved CREATE queue helper returned successfully in read-only mode.
Receipt SHA-256:
`d0472ac3f404a5fb09edb345eb9e3db8ca4f76d78c112c28e5fff42a7e9be5a4`.
No remote jobs, credentials or environment settings changed. The native pilot
and completed local runs establish that this bounded risk-head experiment
fits local memory/time; HPC was not necessary for this work.

Only source, configuration, reports and light aggregate evidence are public.
Raw inputs, caches, checkpoint weights and machine-specific receipts remain
private. The unrelated staged Git changes were preserved with path-only
commits. The Chinese reproduction commands are in `operation_zh.md`.
