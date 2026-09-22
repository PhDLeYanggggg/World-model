# Fraction-Square Control: Execution and Reproduction

2026-09-22. Config, implementation, registration and training scope were frozen
in `a9292047` before fitting. Parent hashes and 1,317 source bindings passed
preflight. Native arm64 `.venv-pytorch`, CPU threads 4, interop 1, workers 0.
No DataLoader multiprocessing, x86 Conda, NumPy fallback or HPC training was used.

## Completed Work

- Full-row 100-update pilot PID 27096 completed. Its original shell session was
  lost during an app transition; completion was checked from the terminal event,
  checkpoint step 100 and PID absence, not an invented recovered shell exit code.
- PID 39835 resumed that exact checkpoint and completed all twelve fixed fits;
  training shell exit 0. Total 144,000 updates, 36,864,000 sampler draws, zero
  unknown-target draws, 123.27743 recorded fitting seconds.
- Checkpoints total 80,138,108 bytes, stored locally and excluded from Git. A
  live sample showed about 3.75 GiB RSS, not a peak-memory benchmark. Four local
  threads and measured runtime did not justify HPC transfer.
- Evaluation PID 40339 and checkpoint replay PID 40488 both exited 0. All twelve
  scores/96 choices replay. Separate verification exited 0 after the fix below.
- 40 scoped tests passed in 2.39 seconds. Prior tests of the same frozen code
  need not be represented as additional independent coverage.

## Verifier Failure Kept Visible

The first separate verifier exited 1. Reconstructing the same algebra through a
different matrix kernel and float64 operation order differed on 5/56,120 first-view
score cells, with maximum violating absolute difference about 1.1641e-5. This
was a verification discrepancy, not silently accepted success.

Only the new verifier was repaired. It now separately checks alternative
geometric algebra and manually assembles the original float32 linear, activation,
fraction and native-cost operations. Every stored score must match exactly.
The regression test includes a prediction-batch boundary and zero disagreement.
Tolerance was not relaxed to make the earlier check pass. No frozen training,
model, threshold, config, choice or analysis file changed.

Separate verification checks 12 fits, 1,581,804 fitting-view rows, 527,268 held
score rows, 96 choices and 768 reductions. Context/smoothness/conditional-quality
calculations are replayed only. This is not independent research replication.

## Commands

From the repository root, using the private local inputs/checkpoints:

```sh
.venv-pytorch/bin/python scripts/run_m3w_fraction_square.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_fraction_square.py --resume
.venv-pytorch/bin/python scripts/run_m3w_fraction_square.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_fraction_square.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_fraction_square.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_fraction_square_head.py tests/test_m3w_fraction_square_verifier.py tests/test_m3w_risk_ranking.py tests/test_m3w_forest_cost_head.py tests/test_m3w_log_cost.py
```

Finished fits are hash-verified and reused by `--resume`; existing receipts are
not overwritten as fresh training. Original private inputs and lineage are
required: the public code alone is not a one-command dataset download/rebuild.

The Git remote was verified at `a9292047` before report updates. The unrelated
3,019 staged paths are excluded from this experiment's explicit-path commit.
Raw data, feature/history/latent caches, checkpoint binaries, third-party data,
images and `.venv-pytorch` are not included. Existing SSH configuration does not
provide a verified CREATE project/connection; remote assets and jobs remain
unverified, not claimed absent. No remote job was submitted or restarted.
