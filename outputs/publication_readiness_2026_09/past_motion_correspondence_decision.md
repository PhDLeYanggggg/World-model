# Moving Past-Frame Correspondence Diagnostic

## Material Passport

Fit-only input diagnostic, not a new forecast model or primary evaluation.
The approved eight-observed/twelve-predicted native-step task is unchanged.
No future labels, development, calibration or confirmation records are used.

## Fixed Procedure

Use the two ETH fit recordings. For each agent take the first contiguous eight
observations with at least 12 pixels net projected movement. Rank agents by
SHA256(recording/id), retain at most 12 per recording, without looking at future
availability, change labels or forecast error. Project using the upstream
row/column convention. Retrieve all eight declared past/current video indices.
No frame-rate rescaling, temporal-offset fit or alternate mapping is selected.

Use a centered 96-by-96 inspection rectangle, no footpoint assumption. Preserve
missing support explicitly. Render point-marked contact sheets locally only.
These rectangles are not detected bodies and markers are not human-gold labels.

On all seven adjacent observed pairs, match an earlier grayscale image template
using zero-mean normalized cross-correlation. Fixed template sizes 15 and 31,
search radius 24 pixels centered at the earlier annotation, with no later
annotation in the matching function. The later observation is used only to
measure correspondence error. Report both sizes, including support failures,
pixel error versus zero shift, peak score and direction agreement. Do not pick
the better size or tune thresholds after seeing results. This is retrospective
within-past correspondence, not prediction of a future step or proof of exact
annotation/sensor-as-of alignment. Small shifts and static background can fool
matching; no global time or geometry certification follows.

## Source Scope

The [ETH Computer Vision Lab data page](https://vision.ee.ethz.ch/datsets.html),
accessed 2026-09-17, specifies research use with author attribution and links the
BIWI annotation/video dataset. This supplies source-side research-use context
missing from the local mirror README; it does not establish a redistribution
license. Do not upload frames/crops/videos or unrelated dataset artifacts.
The original paper PDF search entry was found but its link returned 404; do not
claim to have verified its detailed annotation-point semantics.

No training, primary-metric change, independent-test claim, metric/seconds claim,
Stage5C or SMC. Admission for a predictive visual experiment remains separate.
