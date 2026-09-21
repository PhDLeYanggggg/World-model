# Frozen Cost-Head Diagnosis: Execution

Started 2026-09-20, completed/documented 2026-09-21. Local arm64 PyTorch2.12.0,
four compute threads, one inter-op thread, workers0. Transformer uses the
original CPU device; EqMotion uses the original MPS device with no automatic
CPU/NumPy fallback. No new CREATE job is needed for this readout.

## Commands

```sh
.venv-pytorch/bin/python scripts/audit_m3w_cost_head_fit.py --family transformer --pilot
.venv-pytorch/bin/python scripts/audit_m3w_cost_head_fit.py --family transformer --resume
.venv-pytorch/bin/python scripts/audit_m3w_cost_head_fit.py --family eqmotion --pilot
.venv-pytorch/bin/python scripts/audit_m3w_cost_head_fit.py --family eqmotion --resume
.venv-pytorch/bin/python scripts/verify_m3w_cost_head_fit.py --family transformer
.venv-pytorch/bin/python scripts/verify_m3w_cost_head_fit.py --family eqmotion
.venv-pytorch/bin/python scripts/report_m3w_cost_head_fit.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_cost_fit_forensics.py tests/test_m3w_frozen_risk_forensics.py tests/test_m3w_neural_gain_harm.py tests/test_m3w_frozen_runtime.py -q
```

`--pilot` is for a new directory only. Existing completed runs use `--resume`:
it rechecks source/code/report/array receipts, recomputes diagnostic reductions
and produces no new cost-head forwards. Do not delete results to claim a fresh
run or waive identity checks. Bound runner/module/config bytes remain unchanged.
The verification script replays first/middle/last **original batch boundaries**
of every fold, preserving original device and batch size128. It reconstructs
ADE benefit/harm with independent float64 distance arithmetic, then compares the
original float32 targets exactly. It is explicitly a sampled source replay.

MPS commands require the host's authorized Metal execution outside the restricted
sandbox, as verified in the preceding experiment. This run did not retry a
sandbox failure or switch devices. No Torch resource probing or multiprocessing.

## Completed Sessions

- Transformer included pilot PID31942/session13612 exits0; one seed readout
  0.324855s. Remaining two seeds PID31965/session82488 exit0, 0.460907s including
  pilot-cache checks and completion writing.
- EqMotion included pilot PID32091/session22498 exits0; one seed 0.833463s.
  Remaining two seeds PID32140/session51675 exit0, 1.217426s. Short durations
  reflect frozen small cost-head forwards, not shortened model training.
- Completed resumes sessions95231/76599 exit0: three reused seeds each, zero new.
- Raw verification sessions4772/26625 exit0: Transformer2.038082s and
  EqMotion12.467561s after preflight; 27 batches/2,874 repeated rows each.
- Tests session18397 exits0:45 passed,1 skipped in16.92s. The opt-in MPS unit
  test is skipped in the default suite; actual MPS inference above is separate
  real execution evidence. No full historical writing-integration rerun.
- Reporter exits0 and retains all 12 heads,24 eligibility groups and recording
  slices, including the one small favorable fitting subgroup.

Timing is measured within each recorded operation after preflight, not total
wall time spent inspecting code or loading identities. The pilot heartbeat
retains its intermediate `running` string on early return; the authoritative
terminal session and subsequent completion manifest establish completion. No
process was restarted based on that string. All required sessions are terminal.

## Hashes

| Family | Run identity | Completed analysis |
| --- | --- | --- |
| Transformer | 3bdf0ff03f3053a5933ffc6ecb7dc001131ce32e9732a0dd2ad3a4ef2d9baf11 | 5d7e63098b5705e3c50e9c2e01aa3ac321edc607bfa7f84f24465e1b9429b024 |
| EqMotion | 69b129c0fa8dae0baf9e85a564b09909c091696a7448458ce2e2c09fa8ec3e74 | 2854a2fd50a55ccdd0af7ec2d879deedd3f6212b270f68963bbd536037b0e9a5 |

Private identities, checkpoints referenced by the old runs, per-row readouts and
receipts remain under `data/stage_cvpr2027_experiments/`. Git includes only code,
config, aggregate report JSON/CSV and Markdown. The public verification reports
bind these analysis hashes plus verifier code. No primary metric, policy,
test-use rule or deployable model changed. Stage5C/SMC stay disabled.
