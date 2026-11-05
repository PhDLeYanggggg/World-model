# Validation Receipt: 5 November 2026

## Actual Runs

| Step | PID | Seconds | Observed exit | Evidence |
|---|---:|---:|---:|---|
| Native real-data pilot | 75276 | 35.85 | 0 | pilot_execution.json, pilot.json |
| Six neural fits plus full first refit | 75415 | 491.91 | 0 | train_execution.json, training_freeze.json |
| Held-source evaluation | 76158 | 63.05 | 0 | evaluate_execution.json, analysis.json |
| Independent inference and readout replay | 76300 | 62.21 | 0 | verify_execution.json, verification.json |

The primary fits contain 12,000 actual updates; the independent full refit
adds 2,000. The resume pilot contains another 400 updates including repeats.
All six primary fits finish before held-source evaluation. Training peak RSS
is 7,535,476,736 bytes. Native arm64 CPU4/interop1/workers0, no multiprocessing.

The actual checkpoint audit checks trained common encoder weights, the
zero/nonzero masked/active extra branch, 2,000 optimizer steps, matched
initialization/sampler/targets/preprocessing and the independent full refit.
All three inference, action and metric blocks and all parent controls replay
exactly. This confirms reproducibility, not independent scientific evidence.

Analysis SHA256:
`d9ad2cc040b926429e4321adaa450c51a29697e91d672e7fe336d83521b02da4`.

## Focused Regression Checks

```sh
.venv-pytorch/bin/python -m pytest -q \
  tests/test_m3w_kinematic_risk.py \
  tests/test_m3w_kinematic_feature_provenance.py \
  tests/test_m3w_kinematic_reporting.py \
  tests/test_m3w_direct_tail_risk.py \
  tests/test_m3w_frozen_readout_refit.py
```

Observed exit0: **17 passed in 2.92s**. These include real production encoder
column provenance in a controlled fixture, exact tiny resume, causal-input
rejection, masked-neighbor handling and reporting of failed/undefined views.
The full historical test suite was not rerun.

## Report Replay

Re-executed the unchanged reporter in a fresh native interpreter via its
absolute script path. It reloaded real fitting data, recomputed full fitting
losses and rechecked actual checkpoint/replay hashes. Observed exit0:
**31 pre-existing public artifacts checked, zero changed**. The reporter
uses immutable-write checks; no original metrics or reports were overwritten.
This receipt and the separate next-model note were added after that snapshot.

## Scientific Boundary

The prespecified expansion condition fails; full original risk/support is
0/17. This is one-seed exposed SOURCE development, not a new independent test.
Completion bounds are not sampling confidence intervals. Original 2% rules,
old failures and unavailable primary comparisons remain unchanged. No identity
certificate, deployment upgrade, Stage5C, SMC or submission-readiness claim.

Only code, configuration, reports and lightweight metrics belong in Git.
Checkpoints, caches and source data stay private/local.
