# WEPDTOF source qualification and one-camera causal pilot

Use the Boston University publisher archive and preserve its 14 source
groups: the two Large Office clips share a source, as do the two Jewelry
Store clips. For all clips, audit frame numbers, track-ID uniqueness,
and 8/12 label support at strides 1, 4 and 8. Count all past-eligible
targets; do not require a future label. Treat the annotation masks and
manual track IDs as structural reference, not online observations.

Only Convenience Store is a development camera, chosen from the published
resolution and density table (960 x 720, 6--7 people per frame). Sample
12 evenly spaced, nonoverlapping 20-step intervals at stride 4 from the
archive frame grid. No annotation or future score chooses intervals.
Use the existing frozen COCO MobileNet detector and past-only tracker.
Cache detections, IDs and CV/OLS4 paths from the first eight frames
before matching any label centers. No other clip is used for geometry
matching, training or forecasting. Cold-run the first window.

For the pilot only, match past online and reference centers one-to-one
by distance within half the labeled box diagonal, then require the
same pair across all eight frames. Report lost support and missing
future labels. If at least eight histories in three windows match,
report CV and OLS4 errors in image pixels, their paired observed and
missing-step lower comparison, and a label-history reference diagnostic
on the same matched people. Match and future labels never enter
prediction. No risk certification, independent confirmation, new
formal split, Stage5C or SMC. Released frame steps are not seconds.
