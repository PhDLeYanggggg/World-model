# Reproduction and Runtime

## Frozen Sequence

Diagnostic registration:5b2fb4aa. Diagnostic results and training registration:
28ad1741. All36 final models were frozen and their fingerprints pushed in
5654ce3b before new held scoring. Registration binds the runner,model,readout,
config,tests,protocol and parent evidence. Private model files stay local.

Native arm64 Python3.11 under.venv-pytorch; Torch CPU4,interop1,workers0.
No new CREATE job was dispatched. Checkpoints every100 updates; heartbeat50;
12h boundary;10GiB disk reserve;256MiB private checkpoint cap.

| Phase | PID | Observed status | Duration |
|---|---:|---|---:|
| Frozen-state diagnosis | 184 | captured exit0 | 75.458636s |
| Real resume pilot | 1666 | completion receipt,ended process;terminal exit not captured | 14.109111s |
|36 final fits | 2284 | captured exit0 | 1043.169336s |
|144-view readout | 4207 | captured exit0 | 96.813353s |
|Full144-view replay | 4771 | captured exit0;exact | verified07:05:04 UTC |

Pilot runs80 actual updates across both arms,checking10+10 against20. Its full
training estimate was1339.541s. Actual full training peak RSS is3,130,507,264B.
No training crash or automatic scope downgrade occurred. A shell hash utility
encountered an unsupported locale during reporting; LC_ALL=C fixed that command,
without changing training,artifacts or scientific results.

## Commands

Executed phases,not overwrite instructions for the completed namespace:

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_whole_record_policy register-diagnostic
.venv-pytorch/bin/python -m scripts.run_m3w_whole_record_policy diagnose
.venv-pytorch/bin/python -m scripts.run_m3w_whole_record_policy register
.venv-pytorch/bin/python -m scripts.run_m3w_whole_record_policy pilot
.venv-pytorch/bin/python -m scripts.run_m3w_whole_record_policy train
.venv-pytorch/bin/python -m scripts.run_m3w_whole_record_policy evaluate
.venv-pytorch/bin/python -m scripts.report_m3w_whole_record_policy
```

Verified replay with the retained private assets and authorized source access:

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_whole_record_policy check
.venv-pytorch/bin/python -m scripts.report_m3w_whole_record_policy --check
.venv-pytorch/bin/python -m scripts.report_m3w_whole_record_policy --diagnostic-only --check
```

Incomplete phases accept --resume. Check the saved PID and live process first;
do not duplicate a live run. Completed phases reject overwrite. New experiments
need their own registered namespace. The pre-execution trace-scheduling repair
and its unexecuted draft registration remain documented in the protocol.

## Checks

```bash
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_whole_record_policy.py tests/test_m3w_whole_record_readout.py \
  tests/test_report_m3w_whole_record_policy.py \
  tests/test_m3w_direct_policy.py tests/test_m3w_direct_policy_readout.py \
  tests/test_report_m3w_direct_policy.py tests/test_diagnose_m3w_direct_policy.py \
  tests/test_m3w_shared_train_risk.py tests/test_m3w_shared_train_readout.py \
  tests/test_m3w_unknown_outcome_bounds.py tests/test_m3w_scaled_risk_readout.py \
  tests/test_m3w_component_calibration.py tests/test_m3w_easy_constrained_policy.py \
  tests/test_m3w_policy_contract_v2.py tests/test_m3w_shared_train_report.py \
  tests/test_m3w_selected_tail_audit.py
```

100 tests passed in5.00s. Full historical suite was not rerun. Separate arithmetic
checks5,022 policy summaries,288 frozen controls and888 diagnostic risk means.
Readout and reporter replay exactly. Those are engineering checks,not independent
scientific replication or independent review.

A clean public checkout lacks source data and private checkpoints; anonymous
end-to-end public reproducibility is not yet delivered. Public files contain
code,configuration,aggregate evidence and hashes,not source trajectories,images,
videos,caches or model weights. The source ledger distinguishes fresh_run,
cached_verified and not_run throughout.

Independent roles untouched; exposed TRAIN-only development. Pixel/local
detector-silver,raw annotation steps only; Stage5C/SMC off.
