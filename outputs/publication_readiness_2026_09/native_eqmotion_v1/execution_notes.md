# Native EqMotion Comparator: Live Execution Record

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

The first two phases have run; evaluation/replay/separate-verifier remain
**not_run until all 12 final checkpoints complete**. An active lock/heartbeat is
not completion evidence. The separate verifier has been implemented but has not
yet verified these real results. All three fitting/score phases hold a filesystem
lock, and changed source/config identities are refused. Checkpoints include
optimizer, sampler and Torch RNG; interruptions resume the last atomic state.

The author source, raw data, inputs, predictions and checkpoints stay outside
Git. The public commit contains code/configuration and light reports only.
Four sites remain design-exposed; original closed roles are not opened. No
independent confirmation, calibration, deployment, metric/seconds, Stage5C or SMC.
