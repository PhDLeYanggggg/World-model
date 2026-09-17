# Past-Video Inspection: Available, Not Yet a Validated Input

## Question and Result

The conditional point-forecast comparison reduced false motion but produced no
positive cross-scene trajectory result. The next question is whether the existing
videos can supply past directional information missing from static geometry and
trajectory summaries. This audit tests media access and projection interpretation,
not that predictive hypothesis.

`fresh_run`: the successful v2 run decoded prefixes of both local ETH sources,
retrieving all 62 first-past/current image requests for 31 fit agents. These are
48 distinct frames, not 62 independent observations or eight-image sequences for
every query. The request validator checks eight past/current annotation indices;
the inspection extracts only their first and last images. No new forecast model
was fitted, and no development, calibration or confirmation labels were opened.

| Source | Agents | Image requests | Unique frames | Unclipped inspection rectangles |
| --- | ---: | ---: | ---: | ---: |
| ETH | 5 | 10 | 8 | 10 |
| Hotel | 26 | 52 | 40 | 48 |

The decoder ran for about 3.28 seconds. This is media inspection, not a training
run. The isolated PyAV 18.1.0 arm64 runtime did not change PyTorch dependencies.
Raw images, crops, row identifiers and decoder binaries remain local and ignored.

## Concrete Correction

The local upstream plotting code reverses the inverse-homography output before
drawing: image x is projected column, image y is projected row. The earlier
availability calculation treated that pair directly as x/y. Following the source
convention changes Hotel's in-frame annotation count from 5,417/6,544 (82.78%) to
6,533/6,544 (99.83%). Eleven annotations remain outside. ETH is 8,908/8,908 under
both conventions, so its bounds alone cannot establish the correct convention.

This corrects an image-availability interpretation. The older report remains
preserved, and no frozen native-coordinate forecast, XML context feature, target,
split, primary metric or historical score is changed. Bounds are not proof of
identity correspondence, homography accuracy or metric calibration.

## What the Images Do Not Establish

All six local contact sheets were inspected qualitatively. Some crops contain
nearby people, occluding branches/poles or substantial background. The fixed
96-by-96 rectangle extends 80 pixels above and 16 below the projected point; in
several Hotel examples the person is near its bottom edge. A full rectangle
inside the image is therefore **not** full body coverage. The annotation-point
semantics, individual identity and body orientation are not independently
verified. These observations are neither human-gold labels nor accuracy scores.

Both encoded streams report rate 25. The upstream ETH helper uses 15, while the
documented annotation interval and native spacing also imply a different clock
interpretation from naive encoded playback. Direct-index retrieval follows the
upstream playback convention but does not resolve sensor-as-of alignment or the
physical capture clock. No seconds-level or metric claim follows.

The source README does not establish redistribution permission. No imagery is
published, and this audit supplies no new source-use authorization. Formal
visual-training admission remains `not_run_alignment_and_modality_admission_not_yet_established`.

## Failure, Verification and Next Experiment

The original run failed while serializing NumPy crop coordinates. Its registration,
source snapshot and partial images are retained. The separately registered v2
uses Python integer rectangle coordinates, repeats decoding and completes. Seven
new test cases cover explicit axes, invalid/future frame requests and crop/JSON
behavior; the combined relevant suite passed 18 tests. The full legacy suite was
not rerun. Successful output and local row-record hashes are bound by a completion
receipt; decoding success is not model success.

The next useful work is to resolve annotation-point and frame-index correspondence
using past frames only, and define a source-appropriate crop or scene-context
representation with explicit missing/occlusion masks. Do not silently treat this
inspection crop as a body detector. If admission is established, register a
fit-only trajectory-versus-past-RGB directional probe before increasing model
size. A new modality needs an actual predictive ablation, not merely better
in-frame counts. Independent site support remains a separate requirement.

The eight-observed/twelve-predicted native-step task remains primary; raw-frame
t+50 is supplementary. The pending prospective primary-metric decision is not
resolved by this audit. Historical exposed scenes cannot become untouched tests.
There is no new deployable model, submission-readiness claim, Stage5C execution
or SMC activation.
