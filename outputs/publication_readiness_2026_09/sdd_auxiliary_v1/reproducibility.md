# Auxiliary Comparison Reproducibility

Registration SHA256:
`17d12a071c73a44a42c42c13341c18f4cc4093188b719a4c6a0b57623e8e5fbb`.
Source approval and experiment were recorded before fitting. Commits
`b5b92a5e`, `3e774b5b` and `adb1d5ba` preserve registration, pilot verification
and full input verification respectively.

Runtime: native arm64 `.venv-pytorch`, Torch 2.12.0, NumPy 2.4.6, CPU four
threads, one interop thread, no DataLoader multiprocessing. Atomic checkpoints
include model, optimizer, RNG, phase and sample-draw state every 200 updates.
No GPU fallback, NumPy fitting replacement or full-population-epoch claim.

## Evidence

- All 54 fixed fits completed, 6,000 updates each, including the resumed
  100-update runtime pilot: 324,000 actual optimizer updates.
- Full trainer PID 61978 completed with observed exit code 0. About 3.00 hours
  process wall time; 10,660.60 seconds summed fitting time.
- All 54 checkpoint replays exactly match stored held predictions; replay
  adds zero updates. See [replay.json](replay.json).
- Completed trainer resume PID 78806 exits 0 with zero new updates. All 163
  training identity/checkpoint/prediction/trial artifacts retain identical hashes.
  See [resume verification](resume_verification.json).
- Independent source replay checks 120 frames, 944 crops and 120 geometry/label
  records. All query joins and 600 source cache array hashes were verified.
  This is not exhaustive semantic annotation certification.
- The registered training code/config bindings are unchanged. Only the analysis
  exporter was repaired after fitting for a zero-CV-error event percentage.
  It now reports undefined percentage plus absolute harm, without dropping rows.

## Reproduction Commands

Run from the repository root with the private source assets available:

```bash
.venv-pytorch/bin/python scripts/prepare_m3w_sdd_auxiliary.py --registration configs/m3w_sdd_auxiliary_v1.json
.venv-pytorch/bin/python scripts/verify_m3w_sdd_auxiliary_inputs.py --registration configs/m3w_sdd_auxiliary_v1.json
.venv-pytorch/bin/python scripts/run_m3w_sdd_auxiliary.py --registration configs/m3w_sdd_auxiliary_v1.json
.venv-pytorch/bin/python scripts/run_m3w_sdd_auxiliary.py --registration configs/m3w_sdd_auxiliary_v1.json --replay
.venv-pytorch/bin/python scripts/analyze_m3w_sdd_auxiliary.py --registration configs/m3w_sdd_auxiliary_v1.json
```

The completed trainer is idempotent: rerunning verifies receipts rather than
refitting or selecting a checkpoint. Interrupted runs resume the last atomic
checkpoint. A changed registered source/code binding is rejected, not silently
accepted as the same experiment.

Thirty-six focused tests cover auxiliary loss/masks, causal input isolation,
resume/phase boundaries, source bridges, visual forecasting and the zero-error
report case. The full legacy test suite was not rerun.

```bash
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_sdd_auxiliary.py tests/test_m3w_sdd_step_adapter.py tests/test_m3w_sdd_multimodal_bridge.py tests/test_m3w_offline_visual_forecast.py tests/test_m3w_objective_alignment.py tests/test_m3w_sdd_auxiliary_analysis.py
```

Only code, registration, reports and aggregate metrics are published. Raw source
data, image/history arrays and checkpoints remain local. Public hashes document
identity; they do not replace access to the original licensed data.

## Evaluation Boundary

All 11,966 main fit windows remain included. ETH, Hotel and grouped Zara folds
are previously exposed exploratory physical sites. Three seeds and 2,000
scene-bootstrap draws quantify these results descriptively; they do not create
independent confirmation. Students/development/calibration/confirmation remain
closed for this experiment. No deployment, Stage5C or SMC activation.
