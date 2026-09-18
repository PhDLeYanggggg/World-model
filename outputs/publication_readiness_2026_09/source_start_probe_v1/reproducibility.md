# Reproducing the Source-Supported Start Probe

## Frozen Design

Registration: `configs/m3w_source_start_probe_v1.json`.
SHA256: `bbe433f82cbbe40f393cff61b15a7ddfd9178fd95c4895eceea4a06fe9de7cd2`.
Pre-fit commit: `f0d51451`, pushed before the pilot or held predictions.
The parent registration verifies approved original SDD train40 and every cached
input array. Producer/code/report hashes are bound recursively.

Use the native arm64 `.venv-pytorch` environment on the local machine. The entry
script rejects Rosetta before importing Torch; four compute threads, one inter-op
thread, no multiprocessing DataLoader. Source-only fitting and normalization use
no main inputs or labels. Main-only and mixed fits exclude their held physical
site. All sites are previously exposed **fit roles**, not independent tests.

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_start_probe.py --registration configs/m3w_source_start_probe_v1.json --trial mlp_mixed_seed17_fold0 --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_source_start_probe.py --registration configs/m3w_source_start_probe_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_start_probe.py --registration configs/m3w_source_start_probe_v1.json --replay
.venv-pytorch/bin/python scripts/verify_m3w_source_start_probe.py --registration configs/m3w_source_start_probe_v1.json
.venv-pytorch/bin/python scripts/analyze_m3w_source_start_probe.py --registration configs/m3w_source_start_probe_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_source_start_probe.py tests/test_m3w_source_start_probe_integrity.py tests/test_m3w_observed_unit_frame_v2.py tests/test_m3w_motion_start_probe.py
```

The pilot saves step100 without held evaluation. The main invocation resumes it
to1,000 and runs the remaining fixed matrix; it does not restart the pilot.
MLPs save model, optimizer and sampler/Torch RNG every200 updates. Trees save
every32 estimators and extend the exact seed prefix. Short logistic fits are
atomic final artifacts, not iterative-resume jobs. Completed receipts are checked
before reuse. Do not launch concurrent writers to the same output directory.

`data/stage_cvpr2027_experiments/source_start_probe_v1/heartbeat.json` records PID,
last progress time, current trial and progress. `run.log` retains the full run
stream. If interrupted, inspect the PID/heartbeat first, then rerun the same
registered main command only after the previous process has ended. No code,
source, split or configuration edits are allowed inside an existing run identity.

Reports distinguish `fresh_run` fits from `cached_verified` inputs/artifacts.
The replay and verification JSON files, once present, establish actual completed
checks; commands alone do not. `fit_metrics.csv` contains every prediction cell,
`loss_trace.csv` the MLP losses, and `contrasts.csv` both row-weighted scores and
separately agent-weighted uncertainty. They must not be interpreted as the same
estimand or as independent scene uncertainty.

Private data and model files are prerequisites, not shipped benchmark assets.
No raw data, input caches, history/image stores or checkpoints are committed.
Offline/silver annotation, native steps and dataset-local coordinates remain
explicit. No forecast intervention, metric/seconds claim, Stage5C or SMC.

## Completed Checks

45 probability replays are exact. The completed-resume verifier checks166
immutable model,prediction,checkpoint,trial and identity artifacts,including all
15,000 original Torch updates. It performs zero new fits or updates and preserves
report SHA256`57a82444602b534601138364153ecd107db6d5036bf2cf3431d591661a04879b`.
Seventeen focused tests pass; the full legacy suite was not rerun. A test-only
exact-float equality failed before registration and was replaced with a tight
numeric comparison; no fitted experiment was discarded or changed.

Recorded runtime: Torch2.12.0,NumPy2.4.6,scikit-learn1.8.0. Summed fitting684.70s,
full run11.61min;logistic0.889s,trees677.400s,MLPs6.406s. These are small fixed
classifiers,not a full multimodal model or full-source epoch training claim.
The source MLP uses64,000 sampled rows per fit with replacement. Source-only
and mixed sample masses differ by design; see the registration rather than
assuming equal numbers of distinct source examples across schedules.
