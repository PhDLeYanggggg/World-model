# OpenDD: cross-fit the decision to rotate, not the angle grid

The frozen factored head rotates 710 of 780 edited rows at zero
margin. For each outer location and seed, calibrate its margin on
the other six locations. Each calibration score comes from a head
trained on the other five, excluding both the calibration and outer
location. Reuse a fit only when the two excluded locations and seed
are identical. The calibration recipient menu uses an existing
pair-excluded nominator and a fixed chronological count rule,
not an outer-exposed release plan. Calibrate with thresholds 0,
0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, and keep all original
directions. Margins are in the model's normalized score units.

Mean calibration maximizes conservative gain over the incumbent,
requiring positive mean gain and no fewer total dual passes than the
incumbent in the donor folds. The stable arm additionally requires
nonnegative gain at each donor and activity at two or more locations.
If no nonzero choice qualifies, keep the original. Break ties by
fewer rotations and then by larger margin. Freeze before outer scoring.

Apply thresholds to the already frozen six-donor factored head.
Its training-set size differs from the five-donor calibration heads;
this tests transport of the margin, not a finite-sample guarantee.
On the unchanged 780 edited recipients compare the original, frozen
zero threshold, mean and stable margins, and a donor-constant nonzero
angle on the mean arm's exact rotation mask. Keep the original 2%
risk and utility definitions, native 8/12 steps, and 3,000 paired
location draws after averaging seeds. This development comparison
does not use external calibration or confirmation labels and is not
a release decision.
