# Replaying the main evidence counts

Run `python verify.py` with Python 3.10 or later. No network,
repository, accelerator, checkpoints, video or trajectory data is
needed. The package checks the hashes of its contents, then computes
the counts from minimized frozen result fields.

- OpenDD development: sum the hindsight solver status counts by
  location and the past-only controller counts by location and seed.
  Both used the same nested candidate menu and chronological budget.
  They did not necessarily intervene the same number of times. The
  606 views are 202 recordings observed across three seeds. The one
  unresolved solver result is not a negative certificate.
- MovingDroneCrowd++ development: count recording-seed views where
  the relaxed recording-wide OLS4 utility bound is nonpositive. This
  is a capacity diagnostic, not a score for a learned policy.
- WEPDTOF: count the fixed-count joint proposal's passes against both
  CV and OLS4, using the frozen risk decision and positive conservative
  utility against both references. Groupings are retained without
  filenames. This does not recalibrate risk or rescore a video.

This is an aggregate replay, not an end-to-end reconstruction of
detectors, training, inference or outcome labels. The reference
inputs use image coordinates and their declared normalized utility.
The package contains neither reserved confirmation outcomes nor
author details. Previously published counts can still be recognizable;
the package is not a certification of anonymity.
