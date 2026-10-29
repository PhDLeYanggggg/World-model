# Reproduction

## Frozen Sequence

Registration and implementation:8a398cef. Real pilot/reporting:ffb2b9ee.
All36final fingerprints frozen and pushed in2b71530d before new held scoring.
No model, objective, threshold, dataset split or registered readout was changed
after training started. New models remain private; public receipts bind hashes.

Native arm64 Python3.11, Torch2.12.0, CPU4threads/interop1/workers0. No new CREATE
job. Checkpoints every100updates, heartbeat every50, process lock,12hour limit,
10GiB disk reserve and256MiB checkpoint cap. No resource probing or multiprocessing.

|Phase|PID|Observed outcome|Duration|
|---|---:|---|---:|
|Real pilot|8237|exit0;80updates; both arms exact resume|18.932634s|
|Full training|8629|exit0;36fits;72,000updates|3508.267798s|
|Fixed readout|14011|exit0;144views|99.382036s|
|Full replay|14596|exit0;144views and summary identical|verified2026-10-29T08:39:51Z|

Training peak RSS3,687,137,280B. Pilot peak RSS3,207,233,536B. The pilot estimate
was5648.862s; actual training finished without a scope reduction. Each arm used
3,579,900,000support/query forward row occurrences, not independent samples.

## Commands

The following phases were executed. Completed namespaces reject overwrite:

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_cross_record_policy register
.venv-pytorch/bin/python -m scripts.run_m3w_cross_record_policy pilot
.venv-pytorch/bin/python -m scripts.run_m3w_cross_record_policy train
.venv-pytorch/bin/python -m scripts.run_m3w_cross_record_policy evaluate
.venv-pytorch/bin/python -m scripts.report_m3w_cross_record_policy
```

For unfinished phases only, inspect the saved PID/process lock before adding
--resume. Resume includes optimizer, model, duals, episode sequence and RNG.
Do not submit duplicate jobs or overwrite completed evidence.

Exact verification with retained private source assets and checkpoints:

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_cross_record_policy check
.venv-pytorch/bin/python -m scripts.report_m3w_cross_record_policy --check
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_cross_record_policy.py tests/test_m3w_cross_record_readout.py \
  tests/test_m3w_cross_record_report.py \
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

113tests pass6.06s. Full historical suite was not rerun. Separate arithmetic checks
5,022policy summaries and288frozen controls; reports replay exactly. This was not
an independent human/subagent review or independent scientific replication.

Public checkout alone lacks the authorized source data and private checkpoints;
anonymous end-to-end public reproducibility is not yet delivered. No raw data,
feature/history/latent cache, large model, video, image or environment was pushed.
