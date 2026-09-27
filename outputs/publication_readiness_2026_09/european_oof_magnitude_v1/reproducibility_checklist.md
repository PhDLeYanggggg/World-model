# Reproducibility and Evidence Checklist

## Fixed Design

- Registration commit: `35a05b87`; support commit: `bf1ab738`.
- Inner prediction freeze: `e27681e3`, before fitting magnitude readouts.
- Outer prediction freeze: `1a852ca5`, before new source-held scoring.
- Seeds: 17, 29, 43. Six overlapping producer/controller assignments,
  four held controller localities each, full and motion-only families.
- All new heads use 2,000 updates, width64, original balanced sampling and
  unchanged hyperparameters. No checkpoint selection from held outcomes.
- Every preprocessing step, reference label producer and readout fit excludes
  its held locality. The deepest single-site references also exclude the
  outer held locality. Unknown target rows are not sampled as zero costs.

## Actual Execution

- 144 single-site reference heads and 864 two-site auxiliary heads completed:
  1,008 fresh Torch heads, 2,016,000 optimizer updates.
- 432 cost-only inner controls and all original three-site outer heads are
  cached inputs, not freshly retrained models.
- 432 matched two-slope magnitude readouts fitted; all 144 outer views scored.
- Native arm64 Python and Torch, CPU4, interop1, workers0. Checkpoints retain
  optimizer, preprocessing and RNG states in lossless archives.
- Checkpoints and heartbeat every200 updates; real pilot resumed into the
  registered budget. No interrupted or downgraded full training run.
- CREATE was queried read-only before training; no remote job was submitted,
  modified or cancelled. The old queue observation is not a current status.

## Verification Contract

Run `scripts/verify_m3w_european_oof_magnitude.py` using
`.venv-pytorch/bin/python` from the repository root. Completion is evidenced
by `verification.json`, not by this checklist or an existing checkpoint.
The verifier replays reference lineage, all fresh auxiliary predictions,
all readout fits, all outer scores, 1,728 direct MSE checks, 864 exact raw
control metric matches, deterministic reports/figure and scoped tests.
The full historical test suite is not included in this scoped verification.

The complete ordered execution and recovery commands are in `operations.md`.
Checkpoints, scores, detailed row metrics and licensed inputs stay local.
Public code/configs and hash manifests permit auditing, but do not alone
permit full retraining without the recorded data and parent artifacts.

## Scientific Boundaries

Three seeds are averaged within locality before 3,000 paired locality
bootstrap resamples. Four localities per assignment and overlapping
assignments do not constitute independent large-sample confirmation.
Historically exposed source-held rows remain source development. Independent
selection, reserved risk calibration and final confirmation are unopened.

Primary is expected easy-harm cost MSE, not ADE/FDE or easy degradation.
Source cut drift and one/two/three-site producer-size transport remain visible.
Obs8/pred12 native annotation steps, detector pixels only. No metric, seconds,
human-gold, true3D, foundation or physical-safety claim. Deployment unchanged;
Stage5C and SMC remain disabled regardless of engineering verification.
