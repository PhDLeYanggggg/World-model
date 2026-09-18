# Reproducing the Matched Visual Start Probe

## Registered Experiment

The design was committed and pushed as `db7d30f4` before the full experiment.
Registration: `configs/m3w_source_visual_start_v1.json`.
SHA256: `9edee7d4fe9ef0b97d370ad7804f78ec094102580ef5e7ba12e8e0adf5596af9`.
The registry binds the model, runner, cohort, preprocessing and prior receipts.
It fixes 30 models, 36 prediction cells, three seeds and 60,000 optimizer updates.
The source-only model for a given seed/arm is shared across both held-fit sites.

This is a binary annotation-change information experiment, not a new trajectory
policy. Main forecasting still observes eight and predicts twelve native
annotation steps; its primary metric and closed roles are not changed.
ETH and Hotel are repeatedly exposed fit sites, not independent final tests.
The source admission remains original SDD train40 only, with stride12 and
a +144 raw-frame prediction horizon. No physical time equivalence is assumed.

## Runtime and Resume

Use the native arm64 `.venv-pytorch/bin/python`, CPU four threads, one inter-op
thread and no DataLoader workers. The entrypoint rejects x86_64 on macOS before
importing Torch. There is no automatic NumPy fallback or MPS/resource probing.

Do not start a second writer while the current experiment is live. Check
`data/stage_cvpr2027_experiments/source_visual_start_v1/heartbeat.json` and the
saved process/session record first. Each atomic checkpoint contains weights,
optimizer, sampler and Torch RNG states, exact training IDs, draw counts,
loss/gradient trace and the completed step. Reusing the identical command resumes
at the last completed checkpoint. Never delete checkpoints to force a rerun.

```sh
.venv-pytorch/bin/python scripts/run_m3w_source_visual_start.py --registration configs/m3w_source_visual_start_v1.json
```

The training-only pilot used `--trial past_rgb_mixed_seed17_fold0 --stop-at 100`.
It performed no held evaluation. Those 100 updates are inside, not in addition to,
that model's fixed 2,000-update budget. Reported summed fit time includes them.

## Post-Training Verification

Run these sequentially only after the full training writer has exited:

```sh
.venv-pytorch/bin/python scripts/run_m3w_source_visual_start.py --registration configs/m3w_source_visual_start_v1.json --replay
.venv-pytorch/bin/python scripts/verify_m3w_source_visual_start.py --registration configs/m3w_source_visual_start_v1.json
.venv-pytorch/bin/python scripts/analyze_m3w_source_visual_start.py --registration configs/m3w_source_visual_start_v1.json
.venv-pytorch/bin/python scripts/audit_m3w_source_visual_information.py --registration configs/m3w_source_visual_start_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_visual_start.py tests/test_m3w_source_visual_start_analysis.py tests/test_m3w_source_visual_information.py tests/test_m3w_source_start_probe.py tests/test_m3w_source_start_probe_integrity.py tests/test_m3w_observed_unit_frame_v2.py tests/test_m3w_sdd_auxiliary.py
```

Replay requires exact saved probabilities. The verifier checks all 30 budgets,
15 paired sample streams, identical per-pair normalizers, finite parameters and
logged gradients. Completed resume must add zero updates/fits and preserve 91
immutable artifacts plus the report hash. A current output receipt, not this
command list, establishes that these checks have actually completed.

## Statistical Interpretation

The primary probe contrast is RGB versus the same-schedule coverage-only arm.
Both arms use the same architecture, initialization, sampled training rows and
update budget. Coverage-only zeros RGB before the shared encoder. Training
normalization uses only each schedule's training rows and weights.

Scores average seed losses, not probabilities: no ensemble is introduced.
Row-mean Brier differences and conditional agent-balanced intervals have different
weighting and must be identified separately. The 2,000 resamples describe the
five ETH and 26 Hotel local IDs conditional on fitted models, not uncertainty
over independent new scenes. Contemporaneous agents and overlapping windows
also limit independence. Three seeds do not repair this scene-support limitation.

Constant-prior controls prevent mistaking a useful class-proportion shift for
sample-specific visual prediction. Held labels enter score calculation and
descriptive Brier decomposition only, never inference calibration, checkpoint
selection or a new switching threshold. Pixel variation is not body-state gold.

## Artifact Boundary

Public outputs contain registrations, aggregate metrics, loss traces, scientific
plots and verification receipts. Raw data, cached image arrays, per-row outputs
and checkpoints remain under ignored private directories. Hashes identify those
assets but do not replace obtaining lawful local copies or verifying their roles.

The full legacy test suite is not represented as rerun by these focused checks.
No deployment, independent confirmation, metric/seconds, true-3D, foundation,
Stage5C or SMC claim follows from this experiment.

## Completed Verification Receipt

The full run ended normally after 47.11 minutes, with 2,693.54 summed fit seconds.
The pilot contributed 100 updates and the full continuation 59,900: 60,000 total,
not 60,100. All 30 checkpoint probability replays are exact. The completed-resume
verifier checked 15 matched streams and preserved 91 immutable artifacts and
the report hash with zero new fits or optimizer updates. See `replay.json` and
`verification.json` for actual receipts and `loss_trace.csv` for the recorded
training BCE and gradient norms.

The final focused rerun passed all 34 tests in 2.22 seconds. The plotting-only
tick-spacing repair changes neither training nor scoring. This is
not a full legacy-suite result. Private checkpoints and individual predictions
remain local. No training process from this matrix remains running.
