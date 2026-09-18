# Reproducibility and Execution Receipt

Registration: `configs/m3w_sdd_auxiliary_mechanism_v1.json`.
SHA256: `699c9f6fdb346b828dee4411894b2747a05f541e404285325fa0a93f2174837a`.
Aggregate report SHA256:
`cdbbcece43a678577b97fa6f1d5d702026323b63d751965f50d0dfcccf7780b4`.

Native arm64 `.venv-pytorch`, Torch 2.12.0, NumPy 2.4.6, CPU four threads,
interop one and zero loader workers. This was real local Torch training,
not NumPy fallback or an import-only runtime check. Pilot PID80309 and full
trainer PID80360 both exited successfully; no failed/hung training process.
All 54 fixed new fits complete, 270,000 steps including 100 pilot updates.
The resumed full invocation executed 269,900 additional steps. Summed fitting
time 8,251.3883 seconds; full trainer wall interval about 8,387.85 seconds.

The 54 old fits are hash-verified cached controls, not fresh fits. Original source
train-40 and main-fit caches were reused after hash/schema checks. No source
validation/test admission, sealed-role opening or protocol change occurred.
Offline supplied annotations include retrospective interpolation; no strict
sensor-as-of guarantee is claimed.

## Completed Checks

- All 54 new checkpoint predictions replay exactly on their original held-fit
  rows (`replay.json`, zero optimization updates). Replay PID93083 exited zero.
- All 270,000 recorded steps, finite model parameters/logged losses/gradients,
  exact main draw counts and final sampler states match expectations. Permuted
  source draw counts match the real-source controls. Main4k draws no source rows.
  See `stream_verification.json`; full sampler order is separately unit-tested.
- Completed resume verifies 166 immutable artifacts (54 checkpoints, 54
  predictions, 54 trial receipts, three donor maps and training identity),
  preserves all hashes and adds zero updates. Aggregate report unchanged.
  See `resume_verification.json`; heartbeat/input-check timestamps are mutable.
- Twenty-three focused tests pass, including zero-pretraining exact resume,
  label-only permutation, same main stream and analysis/decomposition checks.
  The full legacy test suite was not rerun.

## Commands

Run from the repository root with the private source/cache assets available:

```bash
.venv-pytorch/bin/python scripts/run_m3w_auxiliary_mechanism.py --registration configs/m3w_sdd_auxiliary_mechanism_v1.json
.venv-pytorch/bin/python scripts/run_m3w_auxiliary_mechanism.py --registration configs/m3w_sdd_auxiliary_mechanism_v1.json --replay
.venv-pytorch/bin/python scripts/verify_m3w_auxiliary_mechanism_run.py --registration configs/m3w_sdd_auxiliary_mechanism_v1.json
.venv-pytorch/bin/python scripts/verify_m3w_auxiliary_mechanism_resume.py --registration configs/m3w_sdd_auxiliary_mechanism_v1.json
.venv-pytorch/bin/python scripts/analyze_m3w_auxiliary_mechanism.py --registration configs/m3w_sdd_auxiliary_mechanism_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_auxiliary_mechanism_analysis.py tests/test_m3w_auxiliary_mechanism.py tests/test_m3w_sdd_auxiliary.py tests/test_m3w_sdd_auxiliary_analysis.py
```

The trainer automatically resumes matching partial checkpoints and skips only
hash-verified complete fits. Do not modify bound code/configuration mid-run or
start duplicate trainers. Future experiments need separate registrations and
output directories. Complete-run checks do not establish independent predictive
validity. Bootstrap is descriptive over three reused sites; no model deployment,
Stage5C/SMC execution, metric/seconds, true-3D or foundation claim follows.
