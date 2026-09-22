# Nested EqMotion Execution Record

## Completed and Verified

Latest: session78894 exited0 with18/18cache producers and12head views.
Fixed-block replay session77078 and separate arithmetic session23799 both
exited0. All1,581,804cost rows and36ordered exclusions pass;5,484fixed rows
replay exactly. This is not full-cache checkpoint replay. The completed evidence
is in`conclusions.md`, `analysis.json`, `verification_with_replay.json` and
`independent_verification.json`. Earlier progress snapshots below are history.

## Training Completed, Cache Pending (Historical Snapshot)

The full training process (session63559/PID23846) exited0 after all18fixed
fits:72,000updates,4,608,000draws and27,903.47 recorded fitting seconds
(7.75hours). Every final checkpoint passed exact sampler/loss-factor matching
to its Transformer counterpart; excluded-site draws total zero. There was no
budget reduction or training restart. The initial100-update pilot was resumed.

The initial cache invocation (session8833/PID56809) was interrupted. A current
process check confirmed that PID absent, and the same registered cache command
resumed in session78894/PID61496. Seven complete producers were verified and
reused; the interrupted producer cache is recomputed because its partial chunks
were not an atomic completion. No model fit was restarted.

Cache completion, checkpoint
replay and the separate real arithmetic audit are not yet established at this
entry. Cost-head fitting remains not_run. Earlier partial snapshots below are
dated execution history, not the current training status.

Registration `fcce08cb` was pushed before real-data fitting. The initial
100-update pilot for coupa/deathCircle seed17 completed in38.49seconds;
6,400 draws cover6,123 training rows, with zero excluded-site draws. This is a
runtime/lineage check, not a completed producer or a measured model gain.
The same fit is resumed, not restarted with an easier budget.

## Initial Live Run, 2026-09-22

Midpoint update:9of18full fits now have hash-verified completion receipts,
36,000updates and13,834.31 recorded fitting seconds, with zero excluded-site
draws. The tenth is active in the same invocation. The lightweight
`training_progress.json` records exact checkpoint hashes without uploading
checkpoints. The matrix and downstream cost-head repair are not complete.

At the first hour checkpoint, two full fits have accepted completion receipts:
8,000 updates,3,046.57 recorded fitting seconds and zero excluded-site draws.
The third seed continues from update1,950. This is partial progress, not a
completed18-fit matrix. A separately written verification program is now added
and its lineage rejection tests pass; the full cost-cache audit is still not_run
because caches require every producer first. The combined scoped suite has47tests.

```bash
.venv-pytorch/bin/python scripts/verify_m3w_eqmotion_nested.py
```

Run this final audit only after cache construction and fixed-block replay. It
independently reduces native errors/costs, checks every pair's fitting weights
and draws, and reconstructs all12 view memberships. Same-agent verification is
not independent research replication. The registered trainer/config is unchanged.

The full18-fit invocation was launched as PID23846, terminal session63559.
An observed training heartbeat advanced to update150 after resuming update100.
These identifiers are historical observations; check live state before deciding
a future interruption or restart. The matrix is not complete at this entry.
Expected fitting time from the pilot is roughly7-8hours, plus inference and
verification. Slow progress is not a failure or permission to reduce the matrix.

```bash
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_nested.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_nested.py --trial coupa__deathCircle_seed17 --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_nested.py --resume
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_nested.py --cache
.venv-pytorch/bin/python scripts/run_m3w_eqmotion_nested.py --verify --replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_eqmotion_nested.py tests/test_m3w_native_eqmotion.py tests/test_m3w_native_nested.py tests/test_m3w_native_forecast.py
```

The first three commands have been invoked. Cache and replay are not_run at
this entry.37 targeted tests pass, including real tiny EqMotion training,
bitwise resume and future-label-invariance checks. Tiny tests do not substitute
for any of the18 full registered fits.

CPU4/inter-op1/workers0/nativearm64 matches the preceding successful real
comparison. Checkpoint every200updates, heartbeat every50updates. The runner
holds a file lock. If interrupted, first verify the process is terminal, then
resume the same command; do not edit registered code/config to bypass identity
checks. A pilot checkpoint is saved at its explicit endpoint as well.

Local free space was about58GiB before launch, sufficient for these bounded
checkpoints/predictions. Current SSH configuration has no CREATE alias; no new
remote scheduler state or asset inventory was obtained. Do not infer from that
that CREATE has no jobs or data. Local data/runtime and the measured budget
justify this local run; no unrelated remote hosts were contacted.

`fresh_run`: real pilot and ongoing new fits. `cached_verified`: prior source
inputs,12outer EqMotion models,18matching Transformer samplers and366bound
dependencies. `not_run`: completed18-fit matrix at this entry, cost cache,
cost-head fitting, independent calibration and deployment. No Stage5C or SMC.
