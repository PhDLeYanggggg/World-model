# SDD Past Images Joined to the Eight-to-Twelve Interface

## Material Passport

An implemented multimodal input path and source-quality diagnostic, not a new
forecast model or a training result. Original SDD train recordings only. The
ETH/UCY primary, 11,966 fit windows and sealed evaluation roles are unchanged.
The research goal is not achieved or submission-ready.

The previous geometry bridge was a prerequisite, not evidence that visual inputs
were available. The old image store covered only the first 64 frames of each
recording. Most actual trajectory queries could not use that store. This work
extracts the registered past requests across the recordings and joins them to
geometry by recording, agent, frame and stride. It closes that concrete gap.

## Scope and Results

Selection is the geometry bridge's fixed 64 evenly spaced past-only queries per
recording/stride, or every query if fewer. Diagnostic strides are 1 and 12 raw
frames. They do not establish equivalent elapsed time or select a training
horizon. The requested model shape is eight observations and twelve predictions.
No future target API is called or target array saved. Raw annotation files contain
complete trajectories, but each query's inputs use its own current/past rows.

| Quantity | Result |
| --- | ---: |
| Original train recordings / physical scene folders | 40 / 5 |
| Registered geometry-image query joins | 5,074 |
| Sequentially decoded source frames | 362,417 |
| Requested frames converted to RGB | 30,798 |
| Unique recording/frame/agent crop requests | 39,144 |
| Crop requests after frame63 | 38,449 |
| Crops with partial geometric image extent | 3,721 |
| Crops intersecting inferred dark-border support | 84 |
| Crops with no retained pixels | 0 |
| Source occluded / generated crop rows | 3,311 / 38,259 |
| Private array bytes, including NPY headers | 323,957,560 |
| Sum of per-record build/check durations | 755.11 seconds |
| Optimizer updates | 0 |

All 5,074 histories have some retained pixels at each of their eight steps.
This does **not** mean every crop is fully supported or every person is visible.
The partial-image counts are retained, not discarded. These are overlapping,
input-selected requests, not independent samples or population prevalence.

The run uses native arm64 Torch2.12.0, NumPy2.4.6 and PyAV18.1.0, one process,
four decoder/Torch threads and Torch interop1. Pilot bookstore/video0 takes
32.20 seconds; the complete invocation reuses its receipt and processes 39 more
recordings. The 755.11 seconds includes parsing, cropping, checks and private
contact sheets, not the later independent replay. It is not model training time
or an epoch benchmark. A process-memory snapshot was unavailable because the
sandbox denied `ps`; no measured peak-RSS claim is made. The live tool handle and
frame/record progress established continued execution, not that denied query.

## What Is Verified

`fresh_run`: image extraction, geometry/pixel joins, input-quality measurement,
independent sampled decode, and a completed-resume verification.

`cached_verified`: raw source hashes, repaired media correspondence, original
train roster, geometry indices and reused pilot/completed image receipts.

`not_run`: auxiliary model fitting, predictive gain evaluation, new source-role
admission, full-cohort image materialization and independent confirmation. The
scientific sampling/source-role choice remains pending. This cache is a sampled
diagnostic, not an image store covering every medium-training query.

The sparse reader refuses uncached history keys instead of replacing them with
zero images. It checks past grid, observation mask, annotation boxes and source
flags against the geometry adapter. Its model-input allowlist contains geometry,
RGB, support and causal CV rollout, not identities or future targets. Pixels are
limited to the query's past frames; masks use the current image only. Original
and retained RGB remain separate, and the border mask is inferred rather than
visible-body segmentation or human gold.

An untrained CNN/MLP's visual features are finite and its zero-initialized skip
output equals CV exactly for all sampled joins. That is an interface check, not
a training smoke test, non-collapse result or downstream lift. Existing runtime
training checks are not replaced by this zero-update check.

Independent first/middle/last requested-frame replay covers 120 frames in all
40 recordings. All 143 crops at those frames match stored RGB and support counts
exactly. This is deterministic source/pipeline replay, not an alternate annotation
or semantic certification. The full set of 39,144 crops is not independently
re-decoded. A completed resume performs zero new extraction and preserves 481
array/metadata/query/contact-sheet/receipt/main-report hashes; the live heartbeat
is intentionally excluded. Twenty-nine focused tests pass, including five new
join/cache-miss tests. The full legacy suite was not rerun.

## Input Quality, Not a Failure-Cause Claim

The short side of the mapped annotation box, projected into a pooled 32x32 crop,
has quantiles [min,p10,p50,p90,max] = [2.43,5.19,9.00,15.55,41.37] pixels.
833 crop requests (2.13%) are below 4 pixels, 15,145 (38.69%) below 8, and 35,802
(91.46%) below 16. A box can exceed the fixed crop; this quantity is not a count
of visible person pixels. The thresholds are descriptive, not exclusion or
training-admission rules. All registered requests remain.

Small, low-contrast actors and shadow/background-dominated crops limit what can
be inferred visually. These measurements do not prove that resolution caused
the earlier ETH/UCY forecasting failures or that a larger encoder/crop would fix
them. Those experiments used other sources. A matched training comparison is
still needed before claiming useful visual or auxiliary-data transfer.

[Limited self-review](visual_review.md) covers nine input-selected histories from
bookstore/video0, deathCircle/video2 and hyang/video7. This is tool-assisted
self-audit, not independent human labels. The other 37 recordings have automated
checks only. Retrospective generated coordinates remain offline annotations;
past-indexed pixel access does not establish strict sensor-as-of causality.

## Reproduction and Next Decision

```sh
.venv-pytorch/bin/python scripts/build_m3w_sdd_multimodal_bridge.py --config configs/m3w_sdd_multimodal_bridge.json --recording bookstore/video0
.venv-pytorch/bin/python scripts/build_m3w_sdd_multimodal_bridge.py --config configs/m3w_sdd_multimodal_bridge.json
.venv-pytorch/bin/python scripts/build_m3w_sdd_multimodal_bridge.py --config configs/m3w_sdd_multimodal_bridge.json --verify
.venv-pytorch/bin/python scripts/analyze_m3w_sdd_multimodal_bridge.py --config configs/m3w_sdd_multimodal_bridge.json
.venv-pytorch/bin/python -m pytest tests/test_m3w_sdd_multimodal_bridge.py tests/test_m3w_sdd_past_images.py tests/test_m3w_sdd_step_adapter.py -q
```

The script resumes completed recordings by verified receipts. An interrupted
unfinished recording is rebuilt; completed records are not silently overwritten
under a new identity. A different source/config requires a separately versioned
output, not bypassing hash checks. Private arrays/images remain outside Git.

Config SHA256: `8bce6b4397b854e31ba40b2500eb3a003d0316cdaea755eec634069904bbf5b9`.
Report SHA256: `0baf077c4c1b67ffa7d645fa8cdb4757c44d86a797457473625e310216c7f7d8`.
Code/check contract was committed before extraction as `74ff40c1`.

The same unresolved auxiliary source-role/sampling decision was recorded after
the motion-information study and the geometry bridge. The independent input
prerequisites are now implemented and verified, with no active process. The next
meaningful comparative fit cannot be registered without settling that scientific
choice. Do not invent another same-site classifier grid or treat a preselected
question option as a user answer. The recommendation remains a separately
registered stride12 SDD auxiliary arm and matched no-auxiliary control, with the
main 8/12 evaluation, sealed roles and primary metric unchanged.

No new deployment, forecast improvement, Stage5C, SMC, metric, seconds-level,
true-3D or foundation claim follows. The remaining evidence still needs a safe
positive predictive contribution, independent calibration/confirmation and the
full method/baseline/ablation/statistical package.
