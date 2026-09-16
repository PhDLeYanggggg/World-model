# DUT Raw Annotation Intake and Causal Verification

Date: 2026-09-16. This is a diagnostic data conversion and source audit, not a forecasting experiment, source-use approval, clean test-set designation or submission-readiness result.

## What Changed

I acquired the author's unfiltered DUT annotations at commit `80b8c746833664cd1e5244fccad79c7f2a7cbe31`, verified each file against its Git blob, and connected the positions to the existing causal recording reader. The allowlist contains 56 raw CSVs, 28 ratio text files, the README and one preprocessing script read only for provenance: 86 files / 24,308,492 bytes. No images, videos, filtered states, weights or runnable third-party pipeline were used. The source and 97.69 MB derived cache remain Git-ignored.

The complete pinned tree contained no path named LICENSE, COPYING or terms. Its README requests citation but does not settle all intended research-use and redistribution conditions. Public retrieval for this diagnostic inspection is not recorded as permission for formal training, reuse or redistribution. No raw or third-party code is committed. The [author repository](https://github.com/dongfang-steven-yang/vci-dataset-dut/tree/80b8c746833664cd1e5244fccad79c7f2a7cbe31) links the [dataset paper](https://arxiv.org/abs/1902.00487); this audit does not claim to have independently validated that paper's calibration.

## Fresh Counts

| Item | Result |
|---|---:|
| Clips | 28 |
| Physical-site groups in the author description | 2 |
| Intersection / shared-space clips | 17 / 11 |
| Pedestrian / vehicle tracks | 1,793 / 69 |
| Original position rows | 457,686 |
| Exact raw10 / raw25 windows | 426,290 / 399,364 |
| Exact raw50 / raw100 windows | 356,454 / 278,407 |
| obs8/pred12 windows | 422,656 |
| Track length min / Q1 / median / Q3 / max | 4 / 119 / 216 / 358 / 890 |
| Within-track consecutive raw-frame differences | 455,824 differences, all 1 |

These are **all-agent, overlapping availability windows with eight past observations**, before quality exclusion. They are not pedestrian-only totals, independent samples, model-selection rows, approved train/test counts or seconds-based horizons. The 1,793 raw pedestrian count matches the author's advertised count. The site grouping maps the `intersection` and `roundabout` clip prefixes to the two described locations; it is not 28 independent locations, and independent capture-session provenance has not been established.

## Annotation Defect: Do Not Silently Deduplicate

Independent full-trajectory comparison found a concrete issue in `dut_intersection_04`: source pedestrian IDs **10 and 11** (internal typed IDs 20 and 22) have exactly the same x/y positions at the same frames **1 through 145**. Both are moving tracks, not a coincident stationary point. Treating both as different pedestrians would double-count an agent and alter density, nearest-neighbor and interaction features.

The cache preserves both source rows for traceability. The audit marks **the entire recording for quality quarantine pending annotation review**, rather than silently deleting a track, changing a split or choosing an interpretation based on prediction scores. This recording has 25,113 points. There are 337,694 raw50 availability windows outside it, but those are still unapproved diagnostic data, not a new usable test count. Formal inclusion requires resolving the annotation or explicitly excluding the clip under the frozen data-quality rule.

There are no byte-identical raw CSVs. Exact complete x/y trajectories plus relative frame sequences produced no matches against **3,112 tracks from 47 previously indexed recordings** (nine canonical plus 38 CITR). This narrow test does **not** rule out partial overlap, rounded/reprojected copies, temporal clipping, identity reassignment or historical predictive exposure. The lack of such matches cannot prove independent confirmation eligibility.

## Coordinate, Time and Causality Boundaries

The [pinned README](https://github.com/dongfang-steven-yang/vci-dataset-dut/blob/80b8c746833664cd1e5244fccad79c7f2a7cbe31/README.md) broadly describes trajectories as converted to meters. However, the [author's filtering script](https://github.com/dongfang-steven-yang/vci-dataset-dut/blob/80b8c746833664cd1e5244fccad79c7f2a7cbe31/scripts/filter_trajectories.py) reads `data/trajectories` and divides its coordinates by each ratio **before** filtering. Thus raw pixels are indicated by the implementation, and the general README statement must not be blindly applied to the raw fields.

The 28 author scale values range from 22.83818 to 28.50195. They are not applied here. The script initializes pedestrian velocity from the next frame and vehicle velocity from up to ten later frames. Its filtered state is therefore not accepted as an official causal input. We read raw pedestrian `x,y` and raw vehicle `x_c,y_c`; all motion features use the current/past positions only. No author code is executed.

The README reports 23.98 FPS; the filtering code uses 23.976. Raw annotations have unit frame increments, but this audit did not read original video metadata or independently verify the annotation/video mapping and scale. Metadata therefore retain **dataset-local, scale unverified, raw frames/observation steps only**. No meter, seconds, physical safety or true-3D claim is added. The existing SDD claims are unchanged.

## Implementation and Verification

- `fetch_m3w_dut_annotations.py`: pinned commit/tree, fixed regular-file allowlist, HTTPS certificate checks, 30 MB cap, per-file blob/length checks, atomic writes and refusal to overwrite changed files. Default mode inspects metadata; `--download` explicitly acquires the allowlist. Third-party code is read for provenance only.
- `m3w_dut_recordings.py`: strict raw schemas, finite integer frame/agent validation, typed clip-local IDs `2*source_id + type`, raw CSV line mapping, memory-mapped arrays and lazy past-only scene input. Missing future labels do not remove currently observed agents. No interpolation, goals, central velocity or future-derived feature is introduced.
- Conversion is atomic per clip. `--resume` verifies source/code/run identity, array hashes and completion receipts; only owned partial directories may be rebuilt. PID and progress go to a heartbeat. All clips remain `diagnostic_only` and outside the formal experiment contract.
- `audit_m3w_dut_intake.py`: separately reads the CSV fields without the converter parser, compares **every one of 457,686 rows**, and recounts raw/step windows using uniform-gap runs rather than the index builder. It records the duplicate-annotation quarantine and narrow cross-cache duplicate search.
- Future-coordinate corruption checks cover one raw50-eligible query in each of 28 clips, **844 agent queries**. Scene membership, causal inputs, types and coordinate transforms remain unchanged; future-label API calls: **0**. This is an input-boundary check, not a complete end-to-end leakage guarantee for a future experiment.

Fresh conversion took 11.496 s. A completed resume reused all 28 recordings with the same identity and reverified every raw row in 2.016 s; the heartbeat adds a small amount to cache size. Fresh and cached reports are separate files. No predictor was fitted or evaluated.

Final targeted regression: **117 passed in 1.49 s**, including 29 new DUT tests. Tests cover filtered/nonfinite/ambiguous input rejection, strict row identity, missing frames, absent future labels, interrupted/owned/unowned recovery, source/code/cache drift, independent recount and detection of duplicate agent annotations. The initial missing-module test failed before implementation as expected. The first independent real audit reached report assembly but failed because `sha256` received a string path; the fix and an audit-output regression are included, and the real audit was rerun successfully. A shell locale failure in `shasum` was resolved by `LC_ALL=C`, without relaxing hashing.

No legacy full-suite rerun was made: it is nonhermetic. Its previously recorded 1,870 pass / 1 unrelated data-lake fixture fail remains unchanged. The unchanged real development preflight was run again: **exit 2, Explicit protocol approval required**. No CREATE connection retry, job, new real-data training, score, policy selection or deployment occurred.

## Reproduce

From the repository root, using the arm64 environment:

```bash
.venv-pytorch/bin/python scripts/fetch_m3w_dut_annotations.py --download
.venv-pytorch/bin/python scripts/build_m3w_dut_recordings.py
.venv-pytorch/bin/python scripts/build_m3w_dut_recordings.py --resume --report outputs/publication_readiness_2026_09/dut_causal_intake/resume_report.json
.venv-pytorch/bin/python scripts/audit_m3w_dut_intake.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_dut_recordings.py tests/test_m3w_citr_recordings.py tests/test_m3w_causal_recordings.py tests/test_m3w_external_source_audit.py tests/test_m3w_recording_lineage.py tests/test_m3w_experiment_contract.py -q
```

The second command requires a new output directory; use `--resume` for an existing compatible cache. Preserve fresh evidence on later runs by passing a new `--report` path. Download reruns verify pinned existing bytes, not a moving default branch. The diagnostic source report does not approve scientific roles.

## Evidence and Remaining Decisions

Machine-readable evidence: [source](source_manifest.json), [fresh build](build_report.json), [cached verified resume](resume_report.json), [independent recount and quarantine](independent_recount.json), [verification receipt](verification.json).

This moves DUT from metadata-only to source-pinned, row-audited diagnostic availability. It does not close the independent-scene deficit: two locations cannot be expanded into 28 calibration scenes, and source terms, previous predictive use, duplicate annotation handling and scientific roles remain unresolved. The shortest next step is to resolve those eligibility decisions alongside the already-pending primary protocol/risk choices, then run the frozen matched comparison. Do not train on a location proposed for future independent confirmation while those decisions are open. No repeated request, automatic role assignment or new risk tolerance was introduced here.

CVPR submission readiness remains **not achieved**. Stage5C and SMC stay disabled; no metric/seconds/foundation/deployability promotion is supported by this work.
