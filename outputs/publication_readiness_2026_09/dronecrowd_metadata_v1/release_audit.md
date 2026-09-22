# DroneCrowd Release Metadata Audit

## Material Passport

Fresh small-file acquisition and structural metadata analysis. No annotation
archive, images or video read; no third-party code executed; no new forecasts.

Official lists: 82 train / 30 test / 112 distinct sequence IDs.
ID partition check: True.
This is clip-ID disjointness, not physical-site independence or full no-leakage.

The release README states that val is sampled from test. These folders
cannot define independent selection/calibration/confirmation roles. Their
actual frame membership has not been read. All scientific roles remain unassigned.

## Converter Semantics

- Source XML frame f maps to image index f+1 for frames 0..299.
- Source agent ID maps to derived MAT ID+1; canonical ID is sequence-local.
- The point is a bounding-box center, not a ground-plane footpoint.
- Outside and occluded records are filtered, which can break continuity.
- The six-column output lacks per-track types, visibility and keyframe provenance.
- Existing MAT is loaded without checking whether its XML source changed.
- These are reviewed-source/pattern checks, not execution of author code.

## Remaining Requirements

Original XML structure, interpolation provenance, camera motion, physical sites,
annotation-time mapping, historical exposure and approved roles remain unresolved.
A raw-XML structural screen is implemented and tested on synthetic fixtures only.
It is not evidence that actual DroneCrowd XML is readable or causally suitable.
No metric/seconds, deployment, independent-generalization or CVPR-readiness claim.
Stage5C/SMC remain off.

[Official source folder](https://drive.google.com/drive/folders/1EUKLJ1WmrhWTNGt4wFLyHRfspJAt56WN). File IDs and SHA256 are in source_manifest.json.
