# Verification and Reproduction

## Executed Phases

| Phase | PID | Exit | Scope | Seconds |
|---|---:|---:|---|---:|
| Source-excluded coefficient fitting and decision freeze |55709|0|72 calibrators, all18 frozen heads|56.675966|
| Outcome reductions |56047|0|72 views, four policies each|2.014076|
| Complete decision/readout replay |56321|0|72 views, all arrays and raw controls|not separately timed|

There were no neural optimizer updates or new remote scheduler submissions.
Peak calibration RSS was2,250,358,784 bytes. Native arm64 `.venv-pytorch`,
CPU4/inter-op1/workers0; the real run used Torch for frozen-head inference.
All listed processes are terminal; a stored heartbeat is not a live-job claim.

The first CREATE inventory attempt reached remote execution but its `squeue`
call exceeded20s. A separate read-only check confirmed the owned directory,
manifest and24 packets. Data streams then completed byte/SHA validation. Queue
state remains unobserved, not evidence that an existing job failed. No auth or
remote resource changes were made.

## Checks

62 scoped tests in10 files pass in2.45s. Twelve new tests cover source/recording
exclusion, calibration source weights, unknown/partial labels, unsupported and
empty policies, unchanged2% budget, no target-label inference argument,
completion bounds and removal accounting. The first readout test fixture used
an invalid easy-reference target; the existing contract rejected it. The fixture
was corrected before registration/real execution; the contract was not weakened.

The complete replay verified every saved array hash, raw prediction hash,
raw policy aggregate, calibrator binding, decision mask, per-recording count,
matched count, paired completion bound and bootstrap summary. The posthoc
failure accounting also reproduces unchanged. Research-state JSON parses;
scoped whitespace checks pass. The private aggregate figure was visually
inspected: labels and intervals are readable, with no overlap. It reports full
support counts, not unconditional safety probabilities.

The full legacy test suite was not rerun because some tests write frozen
historical research artifacts. This is not a claim that all repository tests
pass, and unchanged earlier tests are not relabeled as fresh training evidence.

## Replay Commands

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_policy_conditional_calibration check
.venv-pytorch/bin/python -m scripts.diagnose_m3w_policy_conditional --check
.venv-pytorch/bin/python -m pytest tests/test_m3w_policy_conditional_calibration.py tests/test_m3w_policy_conditional_readout.py tests/test_m3w_policy_conditional_diagnosis.py tests/test_m3w_rational_harm.py tests/test_m3w_rational_harm_readout.py tests/test_m3w_harm_jacobian_audit.py tests/test_m3w_selected_tail_audit.py tests/test_m3w_unknown_outcome_bounds.py tests/test_m3w_component_calibration.py tests/test_m3w_shared_train_risk.py -q
```

On a new authorized reproduction checkout, register and commit the protocol
before `prepare`, then run `eval`. Do not delete completed evidence to call a
rerun fresh. `prepare --resume` and `eval --resume` only continue interrupted
work. Per-view atomic files and per-context receipts prevent silent overwrite;
private heartbeat, process lock and12h boundary remain active.

## Evidence Identity and Storage

- Preregistered commit: `f4035029`, pushed before data execution.
- Registration: `9417171ee022229919717d7d68840024b1c3b985f937984743c225afd2c9f803`.
- Decision freeze: `5a2c5ec5b6c068c7add70d5f910464833918735d91a94db10bb62977ef536517`.
- Summary: `f9bb8ebf16ddc8444f84c859df0f40a27a2631347302be1e7a8576bbaecde332`.

Only source/configuration documentation and aggregate JSON/Markdown are public.
The public experiment is about3.3MB, with no file exceeding1MiB. The26MiB
private decision/label cache, model weights, raw data and figure remain untracked.
Unrelated staged entries retain fingerprint
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.
Safe commits name owned paths only; no add-all/reset/revert.

## Scientific Verdict

Registered gate FAILED seven conditions. More supported views do not compensate
for lost full/matched utility and incomplete safety support. No independent role,
deployment, risk budget or historical evidence status changed. Original TRAIN
remains exposed; the three cost-head seeds are not three new forecaster seeds.
Obs8/pred12, raw stride12, detector-silver pixels/local coordinates. No metric,
seconds, human gold, physical safety, true3D or foundation claim. Stage5C/SMC off.
