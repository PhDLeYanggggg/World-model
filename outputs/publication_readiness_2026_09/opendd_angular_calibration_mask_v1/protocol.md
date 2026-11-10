# OpenDD: isolate the rotation mask from angle ranking

Keep the frozen calibration and its original 780 edited rows.
Compare the learned angle with the donor-only constant on the
same mean-calibrated rotation mask. The direct conservative
contrast is computed from both predicted paths; it is not the
subtraction of their separate bounds against the incumbent.

To isolate placement, keep the constant angle and compare
the learned mask to a uniform subset with exactly the same
number of rotations per recording-seed view. The expected
conservative utility relative to the incumbent is additive:
weight each eligible row by k/n. Report how many views actually
allow a different placement. Also report exact expected utility
for uniform nonzero angles at the learned mask. Do not infer
risk-pass counts for these nonlinear random controls.

Keep the 3,000 paired location draws, original 2% rule, and native
8/12 windows. Reuse frozen weights, thresholds and predictions only;
no new targets, thresholds, or releases.
