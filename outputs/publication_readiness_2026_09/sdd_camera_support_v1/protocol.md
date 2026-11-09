# SDD camera development cohort

I used the 23 clips that belong to the training role of both existing
Stage 23 splits. I did not assign a new validation or test role.
Within a clip I partition the native frame indices into disjoint
blocks of 20. A block is eligible if the same two or more pedestrian
IDs are visible in its eight observation frames. I take four evenly
spaced blocks from the eligible roster. Eligibility uses no future
label visibility or coordinates. This is a visible-history cohort,
not a representative sample of all video times.

Both detectors see the same eight decoded frames. RAPiD uses its
1024-square input and MobileNet uses a 4 by 3 overlapping tile grid.
The two arms change both the detector and the image preprocessing,
so their difference does not isolate the effect of tiling. A tracker
uses only the current detections and its past associations. It emits
one trajectory history for an ID observed at all eight frames.
The online predictions are frozen before matching box coordinates.
No future frame pixels are passed to the detectors or forecasters.

I added two development controls after seeing sparse online tracks,
before matching label geometry. The scale probe uses the first
registered window of the first eligible clip in each scene. It keeps
RAPiD's weights and threshold and changes only the image tiling and
cross-tile suppression. The aerial-pretraining arm uses the same 92
windows as the original two arms, 2 by 2 overlapping tiles, the same
tracker, and a fixed score threshold of 0.30. Its [publisher export](https://huggingface.co/EdgeFirst/visdrone-yolo26-det)
was trained on VisDrone2019-DET. I keep the preprocessing and model
effects together for this third arm rather than attributing their
combined difference to pretraining alone.

Within each past frame I associate camera boxes with publisher boxes
using one-to-one assignment. An edge needs IoU at least 0.10 and a
center separation no greater than half the publisher box diagonal.
Both IDs must match through all eight frames to attach outcomes.
Lost and occluded labels are excluded. Available future positions
are publisher box centers, not metric footprints. The release flags
automatically interpolated rows; I count those separately from
non-generated rows. The latter are not an independent annotation
audit. [SDD release](https://cvgl.stanford.edu/projects/uav_data/) |
[Annotation format](https://github.com/flclain/StanfordDroneDataset/blob/master/README).
[Dataset paper](https://svl.stanford.edu/assets/publications/pdfs/ECCV16social.pdf).

I report the loss of support at detection, continuous tracking,
stable matching and full future availability. A clip meets the
engineering support floor with at least eight complete matched
histories across at least three windows. This is not a calibrated
statistical threshold. Clips within a scene may share acquisition
conditions; in particular, matching frame counts do not prove clips
are independent. The five scenes are the grouping level for future
source-separated development.

The separate motion comparison uses the same matched histories,
targets and masks for camera and publisher input. The four motion
rules share the last observed position as their origin. This isolates
the input difference; it does not evaluate risk on unmatched tracks.
The donor export instead keeps all online histories and writes their
past-only inputs separately from the labels and missingness masks.

All results use image pixels and twelve native frame steps. This
does not change the main prediction task, the original CV-relative
2% risk rule, the external WEPDTOF evaluation or the sealed
DroneCrowd confirmation. No learning, deployment or calibration
claim is made from this cohort alone.
