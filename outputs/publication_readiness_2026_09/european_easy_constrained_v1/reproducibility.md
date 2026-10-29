# Reproduction and Execution Record

## Frozen Inputs

Implementation, tests and protocol were committed in `bb270000` before the real
pilot. The registration SHA-256 is
`dae4cf0938a75c7365e90adecda91ed9b2636abfe72a403f6cfb97d355ffedc4`.
The original risk predictions and the preceding donor blend are required to
match their frozen hashes exactly. Every source packet, reference bank, neighbor
set, query mask and parent result is checked. No model or threshold is selected
after examining this readout.

The analysis reporter was added after fitting. It independently checks aggregate
arithmetic but is not a new independent experiment or scientific replication.
It cannot alter frozen fit receipts. The `readout.json` records its source hash.

## Commands

From `/Users/yangyue/Downloads/World`, the real sequence was:

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_easy_constrained register
.venv-pytorch/bin/python -m scripts.run_m3w_easy_constrained pilot
.venv-pytorch/bin/python -m scripts.run_m3w_easy_constrained run
```

The completed run is immutable. Verify it rather than deleting results to rerun:

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_easy_constrained check
.venv-pytorch/bin/python -m scripts.report_m3w_easy_constrained --check
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_easy_constrained.py \
  tests/test_m3w_easy_constrained_policy.py \
  tests/test_m3w_easy_constrained_runner.py \
  tests/test_m3w_easy_constrained_report.py \
  tests/test_m3w_coupled_support.py \
  tests/test_m3w_coupled_support_runner.py \
  tests/test_m3w_coupled_support_verifier.py \
  tests/test_m3w_coupled_support_report.py \
  tests/test_m3w_unknown_outcome_bounds.py \
  tests/test_m3w_scaled_risk_readout.py \
  tests/test_m3w_component_calibration.py
```

An interrupted, incomplete `run` resumes with `--resume`; a completed one refuses
overwrite. The full replay needs the original authorized TRAIN packets and
local model artifacts. Public aggregate reports do not replace these assets.
This is not yet a self-contained anonymous paper reproduction package.

## Runtime and Evidence

- Native arm64 `.venv-pytorch`; CPU threads4, interop1, worker0. No fallback to
  x86 Conda and no claim of new Torch neural training in this experiment.
- Pilot PID88980 exit0:29.068852s,5,214,224,384B peak RSS;4 views,7,572 queries.
- Full PID89348 exit0:246.603598s,6,590,414,848B peak RSS;72 views completed.
- Full replay PID90177 exit0: all72 groups, actions, predictions and diagnostics
  match exactly; final verified event2026-10-29T05:44:09.713810+00:00.
- `complete.json` binds every group and the summary. Final replay and arithmetic
  observations are recorded in `verification.json`.
- 79 related tests passed in3.46s. The previously verified25 prospective-contract
  tests were reused. Neither count denotes the full historical test suite.
- Separate aggregate calculation and its report replay both exit0, checking
  3,384 policy records and144 parent-control identities.

Atomic group checkpoints, process lock, PID/event heartbeats and the unchanged
12-hour/10GiB-reserve limits remain in the runner. Runtime files are under
`data/stage_cvpr2027_experiments/european_easy_constrained_v1/` and are not uploaded.
The inherited loader only reads/hash-verifies authorized original TRAIN packets;
no new CREATE scheduler task is submitted by these commands. This is not an
audit of current remote scheduler status.

## Research Limits

The same 62,902 query occurrences are repeated across6 contexts,3 existing head
seeds and4 sources per context. There are12 source IDs. Counts are not independent
sample sizes. Inner-held records are discarded before forward inference; outer
development and independent selection/calibration/confirmation remain unopened.
Upstream exposure means the fitting evaluation is not whole-pipeline OOF.

Actual full-policy and count-matched utility are reported separately. The original
2% positive-harm/reference and easy-preservation requirements are unchanged.
Unknown/empty ratios remain undefined where appropriate. No confirmatory CI is
claimed from overlapping, previously exposed fitting observations.

obs8/pred12,rawstride12,pixel/local detector-silver. Stage5C/SMC off. No deployment
upgrade, physical-safety guarantee, metric/seconds, true3D, foundation or
submission-readiness claim. Raw packets, features, histories and model artifacts
remain excluded from Git; only code, registration and aggregate evidence are saved.
