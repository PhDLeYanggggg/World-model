# Reproduction and Execution Record

## Frozen Artifacts

Scientific registration commit:13b87ab3. Execution-codec repair commit:35e49920.
Both were pushed before the repaired real pilot and full training. Original
registration remains immutable; execution_registration_v2.json binds the
original registration, amendment and corrected runner/tests. No protocol change.

Native arm64 Python3.11 under.venv-pytorch; Torch CPU4,interop1,no DataLoader
workers. Each policy has12,385 parameters. Three arms share initialization and
recording draws. Checkpoints every100 steps; heartbeat every50.12h limit,
10GiB disk reserve,256MiB checkpoint cap. No new CREATE scheduler job submitted;
authorized existing TRAIN packets are read and hash-verified.

First real pilot PID93772 exited1 before any optimizer update. Repaired pilot
PID94538 completed400 updates in13.253901s,peak RSS2,748,153,856B,exact interrupted
50+50 versus direct100. The terminal exit code for the repaired pilot was not
captured after context transition; its completion receipt and ended process were
verified. Do not invent a captured exit0 for that pilot.

| Phase | PID | Terminal status | Duration |
|---|---:|---|---:|
| Full54 fits | 95082 | captured exit0 | 198.571095s |
| Frozen144-view evaluation | 95704 | captured exit0 | 102.727252s |
| Full144-view replay | 96257 | captured exit0 | complete06:12:22 UTC |

Full train peak RSS3,119,202,304B. Final model hashes and fit receipts are in
training_freeze.json. Complete inference group hashes and summary are in
complete.json. Private checkpoints remain under data/stage_cvpr2027_experiments/
european_direct_policy_v1; no data or checkpoints are committed.

## Commands

Executed training, not a command to overwrite existing completed outputs:

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_direct_policy register
.venv-pytorch/bin/python -m scripts.run_m3w_direct_policy pilot
.venv-pytorch/bin/python -m scripts.run_m3w_direct_policy train
.venv-pytorch/bin/python -m scripts.run_m3w_direct_policy evaluate
```

Safe replay in this workspace with the verified private assets and source access:

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_direct_policy check
.venv-pytorch/bin/python -m scripts.report_m3w_direct_policy --check
.venv-pytorch/bin/python -m scripts.diagnose_m3w_direct_policy --check
```

Interrupted incomplete phases use train --resume or evaluate --resume. Preserve
the PID/heartbeat and inspect process state first; do not duplicate a live run.
Completed phases refuse overwrite. Recreating a fresh experiment requires a new
explicitly registered artifact namespace, not deletion of frozen evidence.

## Tests and Limits

```bash
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_direct_policy.py tests/test_m3w_direct_policy_readout.py \
  tests/test_report_m3w_direct_policy.py tests/test_diagnose_m3w_direct_policy.py \
  tests/test_m3w_shared_train_risk.py tests/test_m3w_shared_train_readout.py \
  tests/test_m3w_unknown_outcome_bounds.py tests/test_m3w_scaled_risk_readout.py \
  tests/test_m3w_component_calibration.py tests/test_m3w_easy_constrained_policy.py \
  tests/test_m3w_policy_contract_v2.py tests/test_m3w_shared_train_report.py \
  tests/test_m3w_selected_tail_audit.py
```

88 tests passed in3.73s. Earlier new reporter tests caught a path-expression
syntax typo; it was fixed before report generation. The training/readout code
and frozen predictions were unaffected. Full historical tests were not rerun.
Separate arithmetic verifies5,310 policy summaries, including empty controls,
all recording masses and paired bounds. Reports replay exactly. This separate
code check is not an independent research replication or independent reviewer.

Anonymous end-to-end reproducibility is not complete: a clean public checkout
does not contain private checkpoints or source data. Current replay is verified
with the local retained models and authorized source access. No claim of full
public reproducibility or CCF-A readiness is made.

Evidence scope: obs8/pred12 rawstride12,pixel/local detector-silver,exposed TRAIN.
No metric/seconds,human gold,true3D,foundation or physical-safety claim.
Independent roles untouched; deployment unchanged; Stage5C/SMC off.
