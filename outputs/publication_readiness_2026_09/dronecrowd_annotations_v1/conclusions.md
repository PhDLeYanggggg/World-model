# DroneCrowd Annotation Audit: Acquired, Not Yet Admitted

## Material Passport

2026-09-23. `fresh_run`: explicitly authorized official annotation download,
complete ZIP/member inspection, all 112 XML/MAT pairs and 30 text comparisons,
structural window counts, duplicate screens and regression tests.
`cached_verified`: the previous five-file release metadata snapshot and exact
repeat of the completed raw audit. `not_run`: training, forecast-error readout,
risk calibration, independent confirmation, camera-motion estimation, scene
grouping from imagery or any image/video download. No source code was executed.

The author approved independent external scenes first and downloading/auditing
the official annotations. The [authorization receipt](authorization.json) records
that decision. It does not assign training, calibration or confirmation roles.
The former route/download-warning blocker is resolved, not still awaiting a reply.

## Acquisition and Integrity

- Official full-release link: [DroneCrowd repository](https://github.com/VisDrone/DroneCrowd).
- [Official folder](https://drive.google.com/drive/folders/1EUKLJ1WmrhWTNGt4wFLyHRfspJAt56WN),
  file ID `1NeUK0AqgACG1iPiu4rjz3J68Pnsaj5LN`.
- Google Drive's cannot-virus-scan warning was accepted within the author's
  explicit download authorization. This is not a malware-free certification.
- Download size: **43,337,913 bytes**; SHA256:
  `2a000ccf37422dd4e32408d8e5b181e974b07eab2b51c96ad68670513094f668`.
- 255 members: one directory, 112 XML, 112 MAT and 30 `_clean.txt` files.
- Uncompressed member total: 616,567,312 bytes, read sequentially in memory;
  the archive was not extracted. Path, type, size, compression and CRC checks
  passed. SHA256 for each of the 254 files is in [analysis.json](analysis.json).
- No executable, image or video member was present. No raw data is committed.
- The fixed metadata still verifies. Its release statement limits use to
  academic/non-commercial purposes; no broader permission is inferred.

Hashes identify this local snapshot, not an upstream signature or immutable
release tag. The historical metadata-only audit remains unchanged.

## Measured Data Support

These are the **author's release partitions**, not new M3W scientific roles.

| Quantity | Release train | Release test | All |
|---|---:|---:|---:|
| Sequence IDs | 82 | 30 | 112 |
| Declared human tracks | 15,633 | 5,167 | 20,800 |
| Raw XML boxes | 4,689,900 | 1,550,100 | 6,240,000 |
| Visible, non-occluded boxes | 3,629,202 | 1,235,078 | 4,864,280 |
| Tracks with no visible rows | 286 | 75 | 361 |
| Retained-track discontinuous edges | 1,700 | 790 | 2,490 |
| Visible boxes outside nominal image bounds | 6,131 | 1,723 | 7,854 |

Every recording's raw frame range is 0..299. Every declared track has 300 raw
boxes, including outside/occluded padding. Dense raw records therefore do not
establish 300 valid observed positions. Visibility removal creates real gaps;
the audit never fills them. All labels are `human`; agent IDs are recording-local.
Each sequence contains 44..516 declared tracks. The coordinate convention is
image-pixel xyxy head boxes; the official point converter uses box centers, not
ground contact points. No meter/second interpretation has been established.

### Structural History/Future Availability

The table counts overlapping **per-agent candidate queries**, not independent
episodes, people or scenes. It requires a continuous visible/non-occluded raw
run. It is an availability audit, not a constructed/admitted training dataset.
The 7,854 visible out-of-bounds boxes have not been silently dropped or clipped;
quality exclusions would change these provisional counts.

| Observation/prediction indexing | Release train | Release test | All |
|---|---:|---:|---:|
| 8 observed, 12 predicted, raw stride 1 | 3,327,435 | 1,131,342 | 4,458,777 |
| 8 observed, 12 predicted, raw stride 12 | 694,648 | 236,066 | 930,714 |
| 8 observed, raw t+10, stride 1 | 3,358,643 | 1,141,999 | 4,500,642 |
| 8 observed, raw t+25, stride 1 | 3,128,384 | 1,063,498 | 4,191,882 |
| 8 observed, raw t+50, stride 1 | 2,761,641 | 938,245 | 3,699,886 |
| 8 observed, raw t+100, stride 1 | 2,084,431 | 707,315 | 2,791,746 |

History-only availability at K=8/16/32/64 is respectively
4,712,148 / 4,542,619 / 4,212,084 / 3,585,836 queries.
For a contiguous run of length L, observation K, prediction H and stride S,
the count is max(0, L - (K+H-1)S). Even stride-12 counts require the entire raw
interval to stay visible. Neither stride is selected as a new external protocol
here, and equal step counts do not establish equal physical time with SDD.
Effective seconds remain unknown.

## Findings That Affect Conversion

### 1. Six MAT files are not exact equivalents of their XML

All 112 MAT files have the same visible frame/agent identities and row counts
as their XML. Only 106 have exactly matching coordinates. Sequences
`00004`, `00006`, `00007`, `00100`, `00107`, `00110` each differ in one
coordinate scalar, by 4 or 8 pixels. The origin of these discrepancies is
unknown; no version precedence or repair has been assumed. The official
converter prefers an existing MAT cache, so recomputing from XML versus loading
MAT can produce different labels. A future converter must pin its source format
and explicitly dispose of these discrepancies, not silently combine formats.

### 2. The 30 clean text files change box geometry

The text-file sequence IDs exactly match the release's 30 test IDs. After testing
the frame-plus-one/xywh interpretation, **all frame/agent identities match XML**.
However, **263,021 of 1,235,078 rows (21.30%) differ in box geometry**. The
differences affect reconstructed xbr/ybr, not frame, agent, xtl or ytl; the
maximum absolute coordinate difference is 16 pixels. No text file is fully
geometry-equivalent. Therefore a mixed XML-training/text-testing pipeline could
introduce a target-coordinate shift. The name `clean` is not evidence of a
forecasting-compatible format or a preferred ground truth.

### 3. Observation-time provenance remains unresolved

Across all **6,240,000 boxes**, only frame, xyxy, outside and occluded attributes
are present. There are no keyframe, generated or interpolated flags. This turns
the previous hypothetical missing-provenance concern into a measured schema
finding. It does **not** measure an interpolation rate or prove that any specific
input leaks future controls. The pinned VATIC source analysis explains why
unflagged dense XML cannot distinguish direct from interpolated observations.
[Prior source-boundary analysis](../annotation_export_provenance_v1/conclusions.md).

An offline-annotated experiment may be useful under an explicit protocol. It
must not be described as strictly sensor-time-causal merely because the loader
uses past frame indices or backward finite differences. No future endpoint,
velocity feature, goal prototype or model input was exported in this audit.

### 4. Physical sites are not identified

No duplicate XML files or exactly equal >=20-visible-row cross-sequence tracks
were found. This narrow screen cannot rule out different views, temporal offsets,
overlapping recordings or repeated physical sites. The archive contains only
annotations, and the observed official folder has lists, converters and image
archives, not a verified per-sequence physical-site map. The author's 70-scenario
description is not a supplied calibration-site mapping. **112 clips are not
112 established independent scenes.**

The release README says validation images are sampled from test. Those supplied
folders must not be used as independent calibration and confirmation sets.
Camera motion, head-versus-ground-point compatibility, physical grouping and
annotation-time mapping still require evidence. Annotation-only permission did
not authorize downloading the separate 7.72-GB training and 2.63-GB test image
archives observed in the official folder. No image download was started.

## Verification and Reproduction

144 scoped tests passed, including the 18 new archive tests and the existing
metadata, annotation-provenance, intake-admission and experiment-contract tests.
This is not the complete legacy test suite. A full offline re-audit reproduces
the analysis exactly. A separate streaming ElementTree recount, without calling
the primary XML inspector, independently matches 112 files, 20,800 tracks,
6,240,000 boxes and 4,864,280 retained boxes. It also observes zero provenance
flag occurrences. This is a technical cross-check, not independent scientific
confirmation. The full audit took about 47 seconds locally; no GPU was needed.

```bash
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_annotations.py --verify
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_metadata.py --verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_dronecrowd_archive.py tests/test_m3w_dronecrowd_intake.py tests/test_m3w_dronecrowd_verification.py tests/test_m3w_annotation_export_provenance.py tests/test_m3w_intake_admission.py tests/test_m3w_experiment_contract.py
```

Archive path: `external_data/DroneCrowd_annotations/annotations.zip`.
Authorization, source/member hashes, result and implementation hashes are bound
in the light reports. The first within-turn audit was extended with per-column
format-difference counts; its preliminary outputs are retained privately, not
confused with this finalized analysis. No frozen prior experiment was modified.

## Decision and Next Step

Acquisition/audit is complete; independent forecast admission is **not** complete.
This source has substantial structural window support, so lack of annotation
volume is not the immediate limitation. The next necessary evidence is a
physical-site/camera grouping and a source-format/observation-mode disposition,
followed by frozen calibration/confirmation roles before model scores are read.
The follow-up image-acquisition question is still unanswered: sparse frames first,
or the approximately 10.35-GB official train/test archives if necessary, only for
scene/camera auditing. No image acquisition has been initiated.
The agreed 8-observed/12-predicted main task and 2% empirical easy-error condition
are not reopened or silently replaced. Do not tune models on these new labels
before assigning their scientific roles.

No new predictive success, deployment, independent risk guarantee, metric/time
calibration, true-3D, foundation or submission-readiness claim is made. M3W
remains a 2.5D trajectory/world-state research system. Stage5C and SMC remain off.
