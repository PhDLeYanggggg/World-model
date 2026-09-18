# Reproducing the Training-Only Deferral Comparison

## Contract

Pre-fit registration: `configs/m3w_source_cost_deferral_v1.json`, commit
`e91a7a37`, SHA256
`136ac22ca3407c58e3db34ce73f37a9d47b5da86fc16036c04e12badafdb207c`.
The analysis consumer was frozen during training at `ac37cfc6`, after the first
branch's training milestones and before joint analysis. Do not call it pre-fit.
Its SHA256 is
`6377d39eb967a211dccdba3667e80e7741a0d629e6792417b5223be4361ff6a9`.

Six continuations, seeds 17/29/43, two fixed objectives, 8,000 additional updates
each from matched 2,000-step mask-only parents. Total 48,000 fresh updates;
6,000 unique inherited updates. Three completed dense cosine controls are
cached and verified, not retrained. No held or main forecast is authorized.

## Environment and Assets

Native arm64 `.venv-pytorch/bin/python`, Torch 2.12.0, NumPy 2.4.6,
CPU threads 4, inter-op 1, DataLoader workers 0. The inherited entry rejects
Rosetta/x86_64 before loading Torch. The real 100-step pilot belongs to the
registered update budget and completed in 9.112 seconds including the initial
training-population prediction. Import success alone is not runtime evidence.

Registered source caches, original cohort assignment, feature normalizer,
parent optimizer/RNG states and completed dense controls must exist locally
with matching hashes. The runner fails rather than fabricate missing assets.
No raw data, media, caches or weights are included in the Git release.

## Commands

Run from the repository root, using the native environment:

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_cost_deferral.py --registration configs/m3w_source_cost_deferral_v1.json --audit-only
.venv-pytorch/bin/python scripts/run_m3w_source_cost_deferral.py --registration configs/m3w_source_cost_deferral_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_cost_deferral.py --registration configs/m3w_source_cost_deferral_v1.json --replay
.venv-pytorch/bin/python scripts/verify_analyze_m3w_source_cost_deferral.py --registration configs/m3w_source_cost_deferral_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_cost_deferral.py tests/test_m3w_source_modality_continuation.py tests/test_m3w_source_continuation.py tests/test_m3w_source_cost_dynamics.py
```

The audit performs no fit or score. The training command automatically resumes
an existing incomplete branch; check heartbeat/PID first and never start two
writers in the same directory. Model, AdamW, sampler and Torch RNG are restored.
Atomic checkpoint and heartbeat interval is 200 updates. After interruption,
only the unsaved tail may repeat; do not add to the registered budget.

On a completed run the same command verifies and skips all six branches. Replay
checks the 24 saved proposal/score outputs exactly. Analysis reconstructs the
three matched sampling streams, validates all milestone states, recomputes
training metrics and verifies completed-resume immutability. It creates
aggregate CSV/JSON reports and a training figure. It cannot produce new held
forecasts or choose a deployment threshold.

## Limits

The completed execution has 24 exact proposal/score replays, three matched
sampler streams and 290 artifacts unchanged by a zero-update completed resume.
The four core test modules plus `tests/test_m3w_deferral_score_diagnostic.py`
pass 32 targeted tests. The original figure was visually checked. The extra
post-hoc score analysis is reproducible with:

```bash
.venv-pytorch/bin/python scripts/diagnose_m3w_source_cost_deferral.py --registration configs/m3w_source_cost_deferral_v1.json
```

This command analyzes final training labels and changes no policy. Its design
followed observation of the first seed, not a pre-fit registration. All six
models are retained; no result licenses held forecasting under this contract.

Engineering checks and loss reduction do not prove prediction utility. All
milestones, seeds, raw proposals and gated actions must be retained. Cost fitting
uses training labels and jointly changing candidates; it is not independent
risk calibration. Three seed ranges are not confidence intervals over new sites.
Report source stride 12 / +144 raw frames, not audited physical time.

CREATE was not used for this local run. Only an old access blocker is known;
no current remote job or artifact is asserted. The full legacy test suite is
not implied by the four focused test modules above. Stage5C and SMC remain off.
