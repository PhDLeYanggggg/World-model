# Native EqMotion Comparator: Completed Execution Record

## Completed Training, Readout and Verification

2026-09-22 Europe/London: all twelve endpoints completed, with 48,000 updates and
3,072,000 draws. Recorded fit time totals 17,715.45s (4.92h), including the resumed
pilot. Training PID89125/session91836 exited0. Evaluation PID14504/session88296
also exited0 after saving all 527,268 predictions and the fixed outcome summary.
Analysis SHA256: `4e5c3a274fa73b3e63d704732d4e474e7b738c859b5bf198a3f510fb78db5d21`.
Full replay PID18201/session26770 and separate verifier session34654 also exited0.
All twelve checkpoints and 527,268 predictions replay with the same analysis
hash. Separate arithmetic checks cover twelve matched samplers, 1,581,804 repeated
fitting rows and 320 scene reductions, including the paired interval. This is a
second implementation by the same agent, not an independent research replication.
The 35 related tests passed again in 11.21s. A full historical
test-suite rerun was not made: unrelated legacy tests can rewrite research reports.

EqMotion improves average ADE more than Transformer, but both fail easy and
exact-zero-CV protection. See [results and interpretation](conclusions.md).
No training budget, source role, tolerance or deployment policy was changed.

Machine-readable receipts: [analysis](analysis.json), [replay](replay.json), and
[separate verification](independent_verification.json). No required process is
left running. First inference took about 44.53 minutes, and full replay about
45.15 minutes, from their first predicting/replaying heartbeat to final receipt.
These are observed phase timings, not estimates or complete initialization times.

## Preserved Launch Record

2026-09-21. Registration commit `33ecf02b` was pushed before real training.
Preflight verifies 268 source/artifact bindings and 12 planned endpoints. 35
related tests passed; the three new native-EqMotion tests pass again after the
optional source-availability guard. These checks are not prediction results.

## Runtime Evidence

Native arm64 `.venv-pytorch`, CPU4/inter-op1/workers0. Real pilot PID88985
completed 100 updates in 38.87s with finite loss/gradients and an atomic checkpoint.
Loss changed from 13.07077 to 1.63642; this is a training trace, not accuracy on a
held site. The full run resumes that same checkpoint, PID89125, local terminal
session91836. Its first resumed heartbeat is step150, not a fresh restart.
At elapsed 3:34, a live process check reports 354.3% CPU and 1,753,328 KiB RSS.
The step count continues advancing; this is computation, not a stale heartbeat.

Pilot-based estimate: 5-6 hours for 48,000 updates across 12 fits, plus verification
and inference. This is an estimate, not completed wall time. 59 GiB local disk was
available at preflight. A process check found no other active M3W training before
launch. No new CREATE connection, asset read, scheduler query or job was made.
The current local SSH config supplies no CREATE alias; unrelated remote hosts
were not contacted. Slowness alone will not reduce the registered budget.

## Recovery and Verification

```sh
.venv-pytorch/bin/python scripts/run_m3w_native_eqmotion.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_native_eqmotion.py --resume
.venv-pytorch/bin/python scripts/run_m3w_native_eqmotion.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_native_eqmotion.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_native_eqmotion.py
```

At launch, only the first two phases had run. Evaluation, replay and the separate
verifier were correctly recorded as not_run; their subsequent status is recorded
above. An active lock/heartbeat is not completion evidence. All fitting/score phases hold a filesystem
lock, and changed source/config identities are refused. Checkpoints include
optimizer, sampler and Torch RNG; interruptions resume the last atomic state.

The author source, raw data, inputs, predictions and checkpoints stay outside
Git. The public commit contains code/configuration and light reports only.
Four sites remain design-exposed; original closed roles are not opened. No
independent confirmation, calibration, deployment, metric/seconds, Stage5C or SMC.
