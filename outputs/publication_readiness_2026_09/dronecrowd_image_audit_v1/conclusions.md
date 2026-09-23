# DroneCrowd Sparse Image and Background Audit

## Status

2026-09-23. **fresh_run:** authorized official-image acquisition, background
registration and cross-clip overlap screening. **cached_verified:** original
XML/member identities and local image hashes. **not_run:** model training,
forecast-error evaluation, independent risk calibration or confirmation.

The author has delegated routine acquisition, technical review and evidence-based
protocol decisions. No further manual authorization is needed for the present
source work. This changes authority, not the scientific evidence requirements.
See the [authorization receipt](../delegated_research_authorization_20260923.json)
and [offline observation disposition](../dronecrowd_observation_disposition_20260923.md).

## Acquisition

The file IDs were observed in the [official DroneCrowd folder](https://drive.google.com/drive/folders/1EUKLJ1WmrhWTNGt4wFLyHRfspJAt56WN).
The provider serves byte ranges, so only ZIP directories and selected compressed
JPEG members were read. The fixed choice is frames 1, 150 and 300 in every clip,
equivalent to XML frames 0, 149 and 299. Selection uses no forecast score.

| Release folder | Clips | Complete directory JPEG inventory | Downloaded audit JPEGs | Image bytes | Range bytes including directory recheck | Acquisition seconds |
|---|---:|---:|---:|---:|---:|---:|
| train | 82 | 24,600 | 246 | 82,600,891 | 137,692,734 | 548.693 |
| test | 30 | 9,000 | 90 | 28,177,604 | 48,103,466 | 194.519 |
| Total | 112 | 33,600 | 336 | 110,778,495 | 185,796,200 | 743.212 |

The two full archives total 11,113,322,231 bytes but were **not** downloaded in
full. Range payload transfer was approximately 1.67% of that total, excluding
small HTTP pages/headers. No val archive, third-party executable or model weight
was acquired. Images and contact sheets remain in ignored local storage.

All selected JPEGs passed ZIP-member CRC, decoding and 1920x1080 dimension checks.
The complete directory inventories match the official recording lists. Directory
inventories were fetched again after acquisition and matched. The field named
`central_directory_sha256` hashes the canonical JSON inventory of names, CRCs,
sizes and offsets, **not** raw directory bytes or the entire remote archive.
Full-archive SHA256 is explicitly unknown. The public
[source manifest](source_manifest.json) contains hashes/metadata, not images.

## Background Overlap

Every pair of first frames was screened: **6,216 pairs**. SIFT descriptors use
half-resolution images with enlarged visible-head-box masks. Unique descriptor
matches, ratio filtering and robust image-to-image homography estimation precede
the fixed support screen: at least 25 inliers, 35% inlier fraction, and 8% convex-
hull coverage in both frames. Thresholds were set before this full real audit;
they are source-overlap diagnostics, not a trained forecasting policy.

There are **64 overlap-candidate pairs**, producing **68 connected components**.
These are not 68 established independent physical sites. In particular:

| Release-train clip | Release-test clip | Inlier matches | Left/right image coverage |
|---|---|---:|---:|
| 00009 | 00065 | 698 | 88.71% / 88.70% |
| 00010 | 00065 | 565 | 73.80% / 74.51% |

The seven first-frame contact sheets were visually inspected in this run. They
corroborate the common road/building layout in these three clips. This is
self-audited visual evidence, not human-gold site annotation. They must stay in
the same source group for any independent-scene protocol, regardless of the
original release folder. The original official split was not necessarily designed
to establish M3W's new physical-site calibration/confirmation roles.

Visual review also reveals rotated, adjacent and differently illuminated urban
and campus views. A failed first-frame match does not prove a different site.
Therefore assigning the 68 components straight to independent roles would be
premature. Multi-view/partial-overlap checks and conservative ambiguity handling
are the next source task; routine author reapproval is not required.

## Apparent Background Motion

Across 112 clips, 224 start-to-middle/end registrations were attempted. **223**
passed the same geometric-support screen. At a fixed nine-point image grid,
**97** supported pairs show median apparent displacement above 5 pixels and
**5** above 20 pixels, expressed at original image resolution.

| Clip | Compared frames | Median grid displacement, pixels |
|---|---|---:|
| 00002 | 1 to 300 | 20.34 |
| 00011 | 1 to 300 | 22.64 |
| 00018 | 1 to 300 | 29.11 |
| 00028 | 1 to 300 | 21.06 |
| 00104 | 1 to 300 | 21.86 |

Clip 00017's 1-to-300 pair has only 22 inliers and does not meet the support
screen. This is unresolved, not proof of a static camera. The displacements are
**image-registration proxies**, not agent speed, verified camera pose or ground
motion. Head masks do not remove every moving object, and nonplanar structures
can produce parallax. Three frames cannot characterize the whole camera path.
These homographies do not establish metric scale or effective seconds.

## Verification and Reproduction

- 194 scoped tests passed in 3.41 seconds, including 32 new image-intake/geometry
  tests. Synthetic translated textures exercise transform recovery; unrelated
  textures, empty masks, out-of-bounds boxes, ignored HTTP ranges, wrong sizes,
  unbounded reads and unsafe paths are checked. These are engineering tests.
- A separate local pass checked SHA256 and dimensions for **336/336** JPEGs.
- Report/source/implementation hashes match the execution receipt. Reaggregating
  saved pair evidence reproduces the 64 edges and 68 components. This is not a
  second full feature-extraction/registration run or independent site annotation.
- The real image audit took **101.336 seconds** on a single OpenCV thread.
  OpenCV 4.14.0.94 was installed only in the local arm64 environment, without
  changing NumPy/PyTorch dependencies or any CREATE environment.

```bash
.venv-pytorch/bin/python scripts/acquire_m3w_dronecrowd_audit_frames.py --split test
.venv-pytorch/bin/python scripts/acquire_m3w_dronecrowd_audit_frames.py --split train
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_images.py
```

The acquisition command can reuse hash/CRC-verified local frames. The audit
command creates immutable result files and refuses an existing result path;
do not delete evidence simply to rerun. Use a separate versioned audit output
when extending this first screen. The full legacy suite was not rerun because
some historical tests rewrite research reports; no all-suite-green claim is made.

## Consequence

The permission blocker is resolved and source work has materially advanced.
The data is not rejected merely for being different from SDD, but its published
clip split cannot be assumed to supply independent physical scenes. Freeze roles
only after grouping and ambiguity disposition, then run the approved offline
8/12 task with explicit dataset-local sampling and raw t+50 supplemental.

No forecasts, thresholds, learned model, deployment or submission-readiness claim
changed. M3W remains 2.5D, not true 3D or a foundation model. Stage5C and SMC remain
off. No further author audit is required to continue the technical source checks.
