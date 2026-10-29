# Reproduce the Coupled Support Experiment

## Material Passport

Code/protocol registration:dedaecb73ef578bf1abba21c652251bd0368d9f77e785b058d2df26b8a449e30.
Prerequisite ancestry and authorized TRAIN packet hashes are recursively verified.
The original failed experiment namespace is not modified. Native arm64 environment
required;no x86 Conda/Rosetta. No scheduler job submission or authentication change.

## Existing Completed Artifacts

From /Users/yangyue/Downloads/World:

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_coupled_support check
.venv-pytorch/bin/python -m scripts.verify_m3w_coupled_support --check
.venv-pytorch/bin/python -m scripts.report_m3w_coupled_support --check
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_selected_analogue*.py tests/test_m3w_pooled_analogue*.py tests/test_m3w_selection_conditioned*.py tests/test_m3w_local_cost*.py tests/test_m3w_coupled_support*.py
```

The first command reruns inference/retrieval/local fits for all72 views and demands
exact row/summary equality. The other two recheck hashes and separate aggregate
arithmetic/plots; they are not fresh independent experiments.79 targeted tests
pass2.72s on the current implementation;not the entire historical test suite.

## Initial or Interrupted Run

The recorded initial sequence was register,commit,pilot,run. Completed pilot/run
receipts are immutable;do not delete them simply to make a command rerunnable.
For an actually interrupted incomplete run,after checking no process owns the
lock,use `.venv-pytorch/bin/python -m scripts.run_m3w_coupled_support run --resume`.
Existing groups must have matching registration;new code requires a new registered
experiment namespace. Keep previous failures and expensive active processes.

Private heartbeat/events/process.lock:
`data/stage_cvpr2027_experiments/european_coupled_support_v1/`.
Atomic per-view outputs in this public package.12h limit,10GiB reserve,
CPU4/interop1/workers0,numericalthreads4,nearestjobs1. No full cache materialization.
The figure is private and hash-bound by figure_receipt.json;it is not committed.

## Interpretation

Original TRAIN inner-fit only;no held/independent forward or query future use.
Donor outcome use is supervised fitting;donor-record exclusion is not upstream
OOF. No deployable policy,dynamics gain,physical safety,metric/seconds,gold,
true3D/foundation or submission-ready conclusion. Stage5C and SMC remain off.
Verification status and exact replay termination are recorded in replay_receipt.json.
