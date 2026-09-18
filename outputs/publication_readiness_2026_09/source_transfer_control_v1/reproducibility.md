# Reproducing the Matched Source Modality Diagnostic

## Scope and Environment

This experiment observes8 supplied annotation positions and predicts12 SDD
stride12source steps. It is a stationary-history source diagnostic, not the
complete M3W benchmark or an independent confirmation study. Bookstore was
excluded from these fits but explored in earlier experiments. No main selection,
calibration or confirmation role is opened. Stage5C and SMC remain off.

Use nativearm64 `.venv-pytorch/bin/python`, Torch2.12.0 and NumPy2.4.6,
CPU4/inter-op1 and `num_workers=0`. Do not use Intel Conda/Rosetta. A prior
48k-update matched continuation took40minutes locally. The100-update pilot
belongs to this experiment's48000newupdates, not an extra model trial.

## Prerequisites

The source input contracts, media cache, complete stationary cohort, six2k
parent checkpoints, six completed10kRGB continuations and frozen same-arm
classifier probabilities must exist locally with their registered hashes.
The repository deliberately excludes raw data, media, per-query arrays and
model weights. A source checkout alone cannot reproduce training without those
legally obtained assets; the audit fails rather than downloading or inventing
them. Main/source roles and originalSDDtrain40 restriction remain enforced.

## Commands

Run from the repository root. All filenames below are frozen version1 assets.

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_transfer_control.py --registration configs/m3w_source_transfer_control_v1.json --audit-only
.venv-pytorch/bin/python scripts/run_m3w_source_transfer_control.py --registration configs/m3w_source_transfer_control_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_transfer_control.py --registration configs/m3w_source_transfer_control_v1.json --evaluate
.venv-pytorch/bin/python scripts/run_m3w_source_transfer_control.py --registration configs/m3w_source_transfer_control_v1.json --replay
.venv-pytorch/bin/python scripts/verify_analyze_m3w_source_transfer_control.py --registration configs/m3w_source_transfer_control_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_modality_continuation.py tests/test_m3w_source_continuation.py tests/test_m3w_source_cost_dynamics.py tests/test_m3w_source_cost_analysis.py tests/test_m3w_recording_diagnostic.py tests/test_m3w_observed_unit_frame_v2.py tests/test_m3w_source_site_quality.py tests/test_m3w_source_site_analysis.py tests/test_m3w_source_visual_start.py
```

Rerunning the training command resumes the last atomic checkpoint, with model,
optimizer, sampler and Torch RNG restored. Complete branches are verified and
skipped. Heartbeat records PID, timestamp, trial, step, loss and elapsedtime.
Checkpoint interval200updates. Check the existing PID/log before launching a
second process; do not run concurrent writers into this directory. A real
interruption can repeat the unsaved tail since the last checkpoint, not add to
the registered completed update budget.

`--evaluate` requires all six mask controls completed and uses all18fixed
predictors. It cannot train, change thresholds or choose a winner. Repeating it
verifies saved predictions, never produces a new independent test. `--replay`
checks18held forecasts and24newtraining milestones. The analysis consumer also
checks48training milestones across both modalities, four-way matched sampling
and readonly completed resume. Its code hashes were separately recorded before
held-source scoring in `analysis_registration.json`.

## Interpretation

All gains use the same stationaryCV denominator. Full trainingfit and held-source
scores must remain separate. Native annotationpixel and past-normalized errors
are different reporting scales, neither establishes meter calibration. EasyCV
error iszero, so easypercentage isundefined, not0% or a passed2%gate. All2000
recordingbootstrap draws are conditional on seven recordings of one exposed
site. Three seeds do not create new scenes. A fixed0.9probability guard is an
old diagnostic, not a newly calibrated gain/harm policy. Main deployment stays
unchanged regardless of this one exploratory site.

CREATE was not used: only a historical authentication/projectpath blocker was
available, and no live remoteasset/job claim is made. Local stable training is
not a NumPy fallback. Full legacy test suite is not implied by focused checks.
