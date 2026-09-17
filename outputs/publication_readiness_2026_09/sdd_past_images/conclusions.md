# SDD Past Images With Explicit Missing Support

## What Changed

The repaired SDD media links now feed a real, hash-checked past-image reader.
It preserves partial crops, separates image extent from inferred black-border
support, retains lost/occluded/generated annotation flags, and never substitutes
later frames for missing history. This is an input-path implementation and real
source diagnostic, **not** a forecasting result or an expanded training cohort.

The research goal remains active and not submission-ready. The approved main
task is still eight observed and twelve future native annotation steps, with
past-normalized ADE aggregated equally across the registered physical scenes.
The fit set remains ETH, Hotel and grouped Zara. Development, calibration and
confirmation roles remain closed. Raw-frame t+50 remains supplementary.

## Fixed Diagnostic Scope

Before extraction, `configs/m3w_sdd_past_image_audit.json` specified all 60 local
recordings, their first 64 decoded frames, queries 0/7/31/63, and history lengths
8/16/32/64 at source-index step 1. This sampling checks reader boundaries and
startup padding; it is not a representative motion benchmark or an approved
sampling interval for future auxiliary training. Eight SDD raw frames are not
asserted to have the same duration as eight ETH/UCY annotation steps.

The source manifest is bound to SHA256
`e1ece86c223328191c53abeaa576cb1aefc68fe90796d8023ff65c4887de26df`.
The diagnostic config SHA256 is
`12f259eac78a091c4423c3381c3df6aa91155ac286748abb7c568a7cd9f972bc`.
Per-record receipts bind the code, source files, arrays and runtime versions.
The arm64 run uses PyAV 18.1.0 with four decoder threads, NumPy 2.4.6 and
SciPy 1.17.1; it neither imports a model for fitting nor launches a CREATE job.

## Results

`fresh_run`: one pilot recording and 59 additional recordings complete the
60-recording extraction. The full invocation reuses the verified pilot receipt;
it does not decode that pilot twice. Total per-record build time is 81.67 seconds,
including parsing, extraction, checks, local figures and window auditing. This
is not a training epoch estimate.

| Quantity | Observed count |
| --- | ---: |
| Decoded source frames | 3,840 |
| Retained annotation rows | 79,680 |
| Non-lost annotation rows | 44,332 |
| Lost annotation rows, retained with zero crop support | 35,348 |
| Occluded annotation rows | 1,720 |
| Automatically generated annotation rows | 77,636 |
| Non-lost crops partly outside image extent | 4,231 |
| Non-lost crops intersecting suspect border pixels | 302 |
| Non-lost crops with no retained pixels | 0 |
| Local array bytes, including NPY headers | 659,413,440 |

The 302 affected rows occur in bookstore/video0, bookstore/video3,
bookstore/video4, hyang/video4 and Nexus annotation video2. This is the count for
the fixed prefixes only, not an estimate over all SDD frames. The zero count of
fully unsupported non-lost crops is not certification that every actor is
visible, correctly identified or semantically aligned.

The reader exercised 11,176 query/agent/history-length combinations. Of these,
5,801 have all requested annotation rows, 5,368 have a complete non-lost history,
and 5,368 have some retained image pixels at every history step. Seventy-two
requests have no retained image support and remain explicitly masked. Repeated
lengths, overlapping frames and shared actors are not independent samples.

Nexus video3 and video5 have no non-lost annotations in their first 64 frames.
Their rows and source frames remain in the diagnostic; no agent population is
created using later appearances and no later image is borrowed. The Nexus3
startup border occupies up to 81.91% of the frame under the fixed detector.

## Reader Semantics

- `rgb_observed` and `geometric_count` retain all in-image pixels, including black
  observations. Out-of-image padding contributes neither intensity nor count.
- `rgb_retained` and `retained_count` provide a second view excluding only
  four-connected components touching the **current** image edge whose maximum
  RGB channel is at most 8. No future frame or reference-image appearance is
  used to estimate the mask. The threshold is fixed, not selected by prediction.
- Means are computed over supported pixels, so padding does not darken the
  remaining image. Crops keep their original center and are not recentered.
- Isolated black interior regions retain support. A real dark region touching
  the border may nevertheless be flagged. Both views remain available; this
  hypothesis is `inferred_only`, not a ground-truth padding mask.
- Lost boxes supply no image crop or trajectory state. Occluded boxes retain
  observed context with a separate occlusion flag; there is no invented
  pixel-level person segmentation or visible-body mask.
- Short histories are padded and flagged, not dropped. An agent must have a
  non-lost annotation at or before the query to enter the query population.
- Requests outside the decoded prefix, unknown agents, unsupported history
  lengths and unapproved data roles raise errors rather than silently changing
  the cohort. The disk cache contains no future-target array.

## Verification

All 54 focused tests pass: 19 new reader/mask tests and 35 existing source,
geometry and partial-image tests. Synthetic cases distinguish interior black
observations, boundary-connected black, image-edge truncation and lost tracks.
They test future-array invariance, short histories, occlusion, changed hashes
and rejected scientific-role/sensor-as-of claims. The full legacy suite was not
rerun, and these tests do not prove a research hypothesis.

`fresh_run` independent replay reopens all 60 source videos and checks 180
fixed frames (0/31/63). All 2,074 non-lost crops at those frames match every stored
RGB/count array exactly. Source rows, annotation flags and mapped boxes match
the bound annotations. This is an independent decode and deterministic pipeline
replay, not an independent semantic annotation or alternate padding algorithm.

Future-row mutation checks pass for 172 eligible source/query combinations:
zeroing later pixels, boxes and flags and changing later agent IDs does not
change earlier inputs or the earlier agent population. Queries without an
eligible observed agent are not counted as passed mutation tests.

`cached_verified` resume checks reuse all 60 completed receipts with no new
decoding or array writes. They verify 180 source hashes, array hashes, all window
summaries and unchanged metadata/receipts/public audit. See the separate resume
receipt; cached checks are not another extraction.

Four private contact sheets were self-audited: bookstore/video0, hyang/video4,
Nexus video2 and Nexus video3, at frames 0/31/63. Large black borders are captured;
the hyang case also illustrates that tiny dark boundary patches can be ambiguous.
These twelve displayed frames are not human gold or certification of all source
objects. The raw images and figures remain local and ignored by Git.

## Limits and Next Action

Source interpolation is still retrospective. Keeping requested images at or
before the query does not make generated coordinates sensor-as-of observations.
The reader requires explicit offline-annotated mode and rejects a strict online
claim. This audit does not newly certify temporal annotation controls, original
video timing, homography or scale. No metric, physical seconds, true 3D,
foundation-model, Stage5C or SMC claim follows.

These previously exposed SDD recordings are not newly untouched evaluation data.
No new source role has been admitted. The pending auxiliary-training source and
sampling decision must be recorded before fitting; prior lineage remains in
force. This repair also does not explain the negative ETH/UCY forecasting fits,
which used different source images.

The next scientific experiment, after admission and sampling are settled, is a
matched comparison of independent state-change training support with and without
the auxiliary source, retaining the current primary, seeds, sealed roles and
negative results. Merely producing these masks is not evidence that vision will
improve prediction. Current formal contribution and independent confirmation
requirements remain unmet.

Artifacts: [extraction receipt](audit.json), [independent replay](verification.json),
[completed-resume check](resume_verification.json).
