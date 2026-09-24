# Execution and Verification Record

## Completed Work

The frozen scientific files and matrix were committed in
`681f54a4f01bd31c5aa26e761db973196dfe50cf`. This report adds the completed
readout without editing that registration, training code, settings or tests.

| Phase | PID | Completion, UTC 2026-09-24 | Actual work | Source |
|---|---:|---|---|---|
| Pilot | 22606 | 21:12:08 | 100 real Torch updates, checkpoint retained | fresh_run |
| Fixed matrix and controls | 22664 | 21:14:58 | Nine ridge fits, nine neural heads, 18,000 total neural updates; all controls | fresh_run |
| Full metric reconstruction | 22881 | 21:16:24 | Recompute complete metrics from verified saved predictions/decisions | cached_verified |
| Checkpoint inference | 22881 | 21:16:42 | All 18 heads, 4,096 sampled rows each, maximum difference zero | fresh_run |
| Accounting/reporting | Separate local process | After completed verification | Verify identities, old-model readouts, actual matched decisions and event support | fresh arithmetic on cached_verified inputs |

The pilot's 100 updates count inside its head's 2,000-update limit. No head was
stopped because it was slow. All processes above returned successfully; no
unfinished training job is represented as complete. The raw process event log
and resumable checkpoints remain private under
`data/stage_cvpr2027_experiments/european_cv_reference_v1/`.

## Runtime and Cost

Native Apple Silicon `.venv-pytorch`, Torch 2.12.0, NumPy 2.4.6, four compute
threads, one inter-op thread, zero DataLoader workers. Architecture is checked
before Torch import. No accelerator resource probing, multiprocessing or NumPy
training fallback. Each neural cost head has 22,914 parameters, 2,000 updates
and 512,000 sampled training draws. Sampling is with replacement; this is not
512,000 unique trajectories. Unknown-label rows are not drawn for supervised
cost loss, but remain in the evaluation and decision population.

The nine neural fit-loop times total about 16.29 seconds, including the resumed
pilot's measured training time. The full matrix-plus-readout process spans
about two minutes; data assembly, artifact checking, prediction, solver work
and bootstrap are not included in fit-loop time. These are small cost heads
on frozen cached forecasts, not a claim that a full world model was trained
in seconds. Minibatch loss traces are in [training_losses.md](training_losses.md).

The prior local pilot justified local execution; no new CREATE job was submitted.
The earlier CREATE handoff observations were historical, not a fresh scheduler
query. The exact remote M3W directory remains unverified. Protected simulation
directories/jobs, credentials, SSH configuration and authentication were not
changed or scanned to discover data.

## Evidence Checks

- Analysis SHA256:
  `322038b4f8b9b70e88abaebcc6920372612cf457130aedc061bcc9797ddcdbd2`.
- The complete metric reconstruction binds to that analysis.
- Each cost-head replay binds checkpoint and sampled row hashes; 18 exact checks.
- Original forecast error/support records are unchanged for all six seed/head
  views. The same holds for the six original pointwise-policy views.
- All 18 decision receipts pass reconstructed recording/frame query checks.
  Matched controls have equal actual binary intervention counts on every query.
- The accounting helper verifies hashes for source recording/frame/site arrays,
  baseline errors, histories and valid-label masks before its support diagnosis.
- 145 scoped tests pass together. Ten report tests explicitly reject failed
  verification, missing/duplicate replay heads, zero support and approximate
  replay, and preserve undefined contrasts. This is not the full legacy suite.
- The source/code/schema/role identities remain frozen. Unknown labels, all
  declared localities and negative controls are retained.
- The rendered comparison figure was visually inspected; labels and intervals
  are visible. Only its generated SVG is included in the light public package.

Engineering checks do not establish scientific safety. All training-source
folds are development-exposed, and the confidence intervals are conditional
on fitted models that share source data. Reserved selection, calibration and
confirmation recordings are unopened. No deployment or paper-readiness gate
passes merely because the code and artifact checks pass.

## Publication Scope

Commit code, tests, configs already registered, these reports, aggregate JSON
and the generated SVG. Do not commit raw recordings, packed arrays, row-level
forecasts/decisions, checkpoints, media, third-party data or `.venv-pytorch`.
The repository contains unrelated staged work; only explicit paths from this
experiment and README/state updates belong to its completion commit.

No Stage5C execution, SMC, independent confirmation, physical calibration or
formal submission occurred. The overall research goal remains unfinished.
