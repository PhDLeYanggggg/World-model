# Source-Policy Selection Operations

Native arm64 CPU4/interOp1/workers0; no new neural weights, checkpoints or row
caches. Parent models and features are cached_verified; this run computes new
source-validation policy choices and freezes new causal action hashes.

Registration commit: `18ab2f66`. Source-choice freeze: `7d98340d`.
Execute from the repository root with `PYTHONDONTWRITEBYTECODE=1`:

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_policy_selection.py register
.venv-pytorch/bin/python scripts/run_m3w_source_policy_selection.py select
.venv-pytorch/bin/python scripts/run_m3w_source_policy_selection.py replay_select
# Commit source_choices.json before transfer decisions.
.venv-pytorch/bin/python scripts/run_m3w_source_policy_selection.py decide
# Commit decision_freeze.json before outcomes.
.venv-pytorch/bin/python scripts/run_m3w_source_policy_selection.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_source_policy_selection.py replay_evaluate
.venv-pytorch/bin/python scripts/run_m3w_source_policy_selection.py report
```

Existing scientific JSON is immutable and must reproduce exactly. Runtime
receipts have wall time and should be retained, not overwritten to simulate a
fresh run. The private process lock prevents concurrent phases. Heartbeats and
events retain PIDs and source/view progress under the experiment's private
directory. Completed phases are reused, not deleted after interruption.

The parent seal binds the previous 72 full fits and their independent numerical
verification. This experiment replays every source choice and every readout;
evaluation independently checks metric arithmetic and per-query count matching.
`verification.json` records actual completed checks. Scoped tests do not imply
full legacy-suite success, raw reconstruction, independent confirmation or
deployment approval. The separate CREATE job37602475 remains an observation-
blocked replication task; no replacement job is launched here.
