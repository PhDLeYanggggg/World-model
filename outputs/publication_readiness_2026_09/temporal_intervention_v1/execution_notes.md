# Execution Record

## Scope and Registration

The fixed protocol and implementation were pushed as `d5f6d12f` before real
training. Local and remote main matched that commit. The preflight verified
1,199 source bindings. Twenty-one scoped transform, protocol and unchanged
log-loss tests passed before fitting. These are not full-suite checks.

Fresh training changes the two scalar risk heads and their candidate-derived
features/labels, not the frozen trajectory predictors. Nested forecasts and
source exclusions are reused with hashes checked. There is no new final-test
role, independent calibration, threshold search or deployment authorization.
Stage5C and SMC remain off.

## Local Runtime and Pilot

Native arm64 `.venv-pytorch/bin/python`, Torch CPU computation threads 4,
interop threads 1, DataLoader workers 0. Disk showed about 50 GiB available.
Existing SSH configuration has no CREATE alias/project. Remote assets, current
authentication and scheduler state are unverified, not declared absent. The
small downstream-head experiment fits locally without a GPU or data transfer.

The real `coupa_seed17/ramp` pilot (PID 19597) completed 100 updates, saved its
checkpoint and exited 0. Loss at step 1/100 was 0.720530/0.406190; gradient norms
were finite. That alone proves neither convergence nor downstream improvement.

The full matrix was started with `--resume` (PID 19637). The first new event
was step 200, continuing the saved pilot rather than restarting it. A live
sample at elapsed 43 seconds showed 123.3% CPU and RSS 3,640,352 KiB, about
3.47 GiB. This is one sample, not a peak-memory measurement. Checkpoints save
every 500 updates; heartbeats every 100 updates. The ignored private run directory
contains events and checkpoints. SIGTERM requests interruption; recovery uses
the last atomic checkpoint with model, optimizer and sampler state checked.

At this record's initial creation, full fitting was still active and readout
had not run. Terminal receipts and the completion section, when present, are
required before claiming completion. The fixed registration is not modified.

## Commands

```sh
.venv-pytorch/bin/python scripts/run_m3w_temporal_intervention.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_temporal_intervention.py --view coupa_seed17 --arm ramp --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_temporal_intervention.py --resume
.venv-pytorch/bin/python scripts/run_m3w_temporal_intervention.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_temporal_intervention.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_temporal_intervention.py
```

Do not rerun the pilot without `--resume` after a checkpoint exists. Do not launch
a duplicate runner while its live session/process holds the run lock. A stale
heartbeat alone is not proof that a process stopped. Recheck the process or
session before resuming.

## Terminal Completion and Verification

The full matrix (PID 19637) exited 0 with 24 completed heads, 288,000 updates,
73,728,000 sampled draws and zero unknown-label draws. Recorded head-fitting time
was 241.35955 seconds, including the resumed pilot once. Evaluation (PID 20250),
checkpoint replay (PID 20346) and the separate formula verifier all exited 0.
The primary scientific gate failed; successful execution does not imply success
of the hypothesis or justify a new deployment.

Replay checked 24 endpoints and 1,054,536 cost scores. Separate formulas checked
3,163,608 fitting-arm row instances, 84 policy choices and 672 scene reductions.
The prefix/interaction diagnostics are replayed, not separately reimplemented.
The subsequent frozen-choice audit ran twice with identical immutable output,
including separate arithmetic for the gates. Eight new diagnostic tests passed;
the preceding 21 unchanged scoped checks are reused. No full legacy-suite claim.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_temporal_intervention.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_temporal_diagnosis.py
```

Analysis SHA256:
`d89afb790c9846ed6519b970a89e5f4b5eb8fc5325b8d082b90fccdde0e3d4b8`.
Action/choice diagnosis SHA256:
`2db314ad7dc40ff34f13303cce771c4c154a121b3fb048b52e62fea7929a6617`.

All required experiment sessions have terminated. Checkpoints, histories,
row-level forecasts and decisions remain in the ignored private data directory.
Public aggregate JSON contains source bindings and summary evidence, not training
arrays or weights. The frozen config, registration and experiment code were not
changed. The additional read-only diagnostic has its own code/test hashes.
The existing unrelated staged-work fingerprint remains
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.
