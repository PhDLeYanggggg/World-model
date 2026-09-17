# SDD Media Correspondence Repair

## Material Passport

Source audit and diagnostic geometry repair, 2026-09-17. No new forecast model,
data role, sampling stride or primary metric is fitted/selected here. Result
sources are separated below. The research goal remains active and not
submission-ready; repairing inputs does not prove a forecasting contribution.

## Full Local Audit

`fresh_run`: decode every frame of all 60 local videos using arm64 PyAV 18.1.0,
four decoder threads and one process. The completed audit covers 522,497 frames
and 10,616,256 annotation rows. Per-video audit time totals 346.84 seconds, including
the pilot resumed into the complete run. No decode errors, missing/non-increasing
PTS or changing frame dimensions were found. Headers match decoded counts 60/60.
Every annotation file matches the local OpenTraj mirror byte-for-byte; this is
local source agreement, not two independent datasets or original-publisher
checksum authentication.

`cached_verified`: a source-hash/receipt resume checks all 60 completed records,
performs no new decode and preserves the report. It is not a second full decode.
Resize verification independently re-decodes 84 sampled frames, and the Nexus link
check replays 60 frame samples exactly against the full audit's pixel hashes.
There is overlap between those two sample sets; do not sum them as independent
observations. Full pixel checksums of every frame were not stored.

## Defect 1: Annotation and Video Pixels Differ

54/60 videos have different dimensions from their associated reference image.
The compressed video is often roughly 55% of the annotation/reference dimensions.
Using native annotation boxes directly on the smaller decoded image can therefore
crop background or the wrong agent. For deathCircle/video2, the reference is
1436x1959 while the video is 792x1080; the correct image-only scale is
(0.5515320,0.5513017), not identity.

The new `SDDImageCoordinates` helper maps continuous annotation box edges to
video pixels, without modifying trajectory labels, changing units or clipping
partially visible boxes. Its past-box API rejects future/invalid frame requests.
Callers must also check decoded-frame availability and visibility; a coordinate
transform alone is not a complete causal image reader.

The mapping changes the number of nominally visible annotation rows whose boxes
fall outside the video extent from 3,217,812 to 148. Coordinate round trips have
maximum absolute error 2.28e-13 annotation pixels. This is structural geometry
evidence, **not** proof all boxes locate the right person. Importantly, the
same-name resize checks alone would miss Defect 2.

## Defect 2: Nexus Same-Name Files Are Mispaired

The original same-name pairing covers every annotation index for only 57/60 videos:

| Annotation key | Annotation max frame | Same-name decoded count | Rows outside decode | Visible rows outside decode |
| --- | ---: | ---: | ---: | ---: |
| nexus/video2 | 12680 | 11472 | 38732 | 23813 |
| nexus/video6 | 12015 | 1062 | 113540 | 78589 |
| nexus/video7 | 12015 | 1062 | 148374 | 95726 |

That affects 300,646 annotation rows, including 198,128 flagged visible. Several other
Nexus files are long enough but show the wrong content, so range coverage alone
is an inadequate admission check.

A fixed filename-order hypothesis explains the pattern: enumerating the original
names lexicographically as video0,video1,video10,video11,video2,... and writing
numeric output names changes the media identity. The resulting explicit links:

| Annotation video | Compressed media video |
| --- | --- |
| 0 | 0 |
| 1 | 1 |
| 2 | 4 |
| 3 | 5 |
| 4 | 6 |
| 5 | 7 |
| 6 | 8 |
| 7 | 9 |
| 8 | 10 |
| 9 | 11 |
| 10 | 2 |
| 11 | 3 |

The hypothesis was fixed from filename ordering and count mismatch, not chosen
by a forecasting score. With it, all 12 Nexus pairs cover the annotation ranges.
On the 10 changed links, median correlation between resized reference and first
decoded frame changes from 0.1344 to 0.9743. Some first-frame correlations stay low
because the image has large black/warped margins, so correlation is supporting
evidence rather than an automatic semantic certificate. The actual historical
compression script has not been inspected; its precise behavior remains an
inference, not a verified provenance fact.

The diagnostic manifest applies this mapping only to Nexus and uses a distinct
video path/hash for each annotation recording. It preserves source IDs and all
raw files. No source video or annotation was renamed, rewritten or uploaded.
All 60 annotation frame ranges are covered under these links. This is a repaired
diagnostic correspondence, not a new official split or model result.

## Visual Review and Remaining Limits

Self-audited contact sheets were inspected for bookstore/video0, deathCircle/video2,
hyang/video7, and Nexus annotation videos 3, 6, 10, each at five fixed frame indices.
The corrected overlays support the resized coordinate mapping and the three
inspected Nexus links. These 30 frames are not a human-gold annotation audit or
certification of all 60 videos. Private figures and original pixels remain local.

The review also finds genuinely missing visual support: black/warped border areas,
occlusion and partial boxes. Bounds-only masks would mislabel some padded areas
as real observations. Neither source flag nor high background correlation proves
person identity or accurate motion. These need explicit masks and supported
past-window checks before visual forecasting. 98.388% of source annotation rows
are marked automatically generated. Any future use must retain the existing
offline annotated-observation disclosure, not claim strict sensor-as-of inputs.

The short clips flagged in the initial inventory are not alone evidence of
truncation: their annotation ranges match their decoded lengths. Conversely,
same-name count equality did not establish Nexus image identity. These two
negative controls are important to the revised source checks.

## Scientific and Licensing Boundaries

SDD has eight named scenes according to the [dataset authors](https://cvgl.stanford.edu/projects/uav_data/),
whose page specifies CC BY-NC-SA 3.0. This audit does not replace the original
license with a third-party mirror's metadata. Dataset content is not committed.
Video counts are not independent calibration-scene counts, and all prior SDD
exposure must remain in the lineage record.

An `estimated_scales.yaml` asset exists in the OpenTraj SDD folder. Its README
explicitly distinguishes estimates, including rational guesses. It is not
verified metric calibration; no scale is applied here. File PTS and rate 2997/100
describe the compressed container, not a newly verified physical clock. No
metric/seconds, true 3D, foundation, Stage5C execution or SMC claim follows.

The preceding 36 negative forecasting fits used ETH/UCY images, not these SDD
videos. This discovery must **not** be relabeled as their demonstrated failure
cause or as a newly positive model result. Historical Stage26/37 scores remain
exploratory for the separate lineage/test-selection reasons already documented.

## Next Step and Gate Status

| Requirement | Status |
| --- | --- |
| Full local decode and hashes | Completed, fresh_run |
| Original same-name index/source correspondence | Failed; preserved |
| Diagnostic resize and Nexus link repair | Implemented and structurally verified |
| Every actor/past-window semantically aligned | Not certified |
| Partial/padded/occluded visual support reader | Still needed |
| New auxiliary-training source role and stride | Pending scientific registration |
| New SDD training or forecasting improvement | not_run |
| Independent paper-level confirmation | Not achieved |

All 35 focused tests pass, including source-index limits, duplicate rows, interpolation
provenance, future rejection, mapping round trips, a synthetic crop-target check
and lexical-link inventory. The unrelated full legacy suite was not rerun.
The public manifest explicitly prohibits treating itself as a training
registration. Keep the current 8-to-12 primary task, fit/dev/calibration/confirmation roles
and raw-frame t+50 supplement unchanged. Before training, register auxiliary source use,
sampling/visibility rules and independent state-change support; do not silently
equate eight SDD frames with eight ETH/UCY annotation steps in physical time.

Artifacts: [full audit](audit.json), [resize checks](resize_mapping.json),
[Nexus correspondence](nexus_identity.json), [diagnostic links](diagnostic_media_links.json),
[cached verification](verification.json), [link verification](link_verification.json).
No new model or threshold was trained.
