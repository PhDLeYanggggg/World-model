# Reproduce the Frozen-Action Diagnostic

Run from the repository root in the native arm64 environment:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python scripts/run_m3w_unknown_outcome_bounds.py replay
PYTHONDONTWRITEBYTECODE=1 .venv-pytorch/bin/python -m pytest tests/test_m3w_unknown_outcome_bounds.py tests/test_m3w_source_policy_selection.py -q
```

The initial `register` phase was committed before the real-data `run`. The
`report` phase creates the final verification seal after replay; completed
immutable receipts must not be overwritten to change results. Raw source data,
the previous local model/checkpoint cache and the source-control manifests are
required locally and are not included in Git.

The runner verifies the committed parent seal, source partitions, row IDs,
prediction/action hashes, old selections and causal envelope alignment. Replay
recomputes every bound and checks exact aggregate artifact equality. The
additional scalar implementation uses independent summation for seven numeric
quantities per candidate. Synthetic tests exercise arbitrary partial masks and
denominator edge cases; they do not establish empirical safety.

No training, checkpoint allocation, new row cache or transfer-outcome readout.
CPU4/inter-op1/DataLoader workers0. A process lock prevents duplicate local runs;
heartbeat and event records are under the ignored private experiment directory.
The planned aggregate output allowance is 8 MiB. No training disk-reserve guard
is changed or bypassed; this run uses the separate aggregate-only allocation.

The existing CREATE job37602475 is an unrelated frozen-boundary replication.
Its latest read-only inspection timed out. That is an observation failure, not
job completion or terminal failure. No duplicate job, restart or cancellation
was performed. The local diagnostic here is complete independently of CREATE.

Full legacy test suite: not_run. New policy choice, new transfer readout and
independent confirmation: not_run in this diagnostic. Do not promote it from
engineering verification to a model-performance claim.
