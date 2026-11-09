# Screen stronger initial forecasts, not a new exchange ranker

Use the registered nine-source Moving fitting-development rotations from
the causal spline experiment, with three seeds per rotation. Train two
matched CV-anchored spline heads with a learned bounded amplitude. Both
read only eight past steps and the existing causal neighbor tokens.
The OLS-only control optimizes the full-grid missing-step-debited OLS4
gain. The dual head optimizes the worse of the CV and OLS4 gains plus
positive excess over the original all/easy/signed 2% risks. The utility
term is query-balanced; the risk surrogate uses unweighted rows.
Each minibatch draws from one training recording, so loss aggregation
does not offset harm across recordings. The soft support incentive and
minibatch risk are optimization devices, not a new evaluation rule.

Compare against CV, OLS4, OLS8, frozen EqMotion, PCT and the prior
completion spline. Evaluate interpolation with CV at the same fixed
amplitudes .02, .10, .35, .70 and 1.0 for analytic and fold-trained paths.
Use only the calibration sources to choose a candidate. EqMotion and
PCT are held-only comparators because their available calibration-source
forecasts are not proven to exclude the outer held groups. For a strict choice require
all calibration recordings to pass original risk and both source-macro
conservative gains to be positive; otherwise select CV and do not
count its undefined edited support as a pass. Also retain the best
diagnostic choice by number of joint passes, then conservative maximin
gain and stable grid order. This diagnostic is not deployment.

Freeze all raw and selected trajectories before held scoring. Report
all seeds and source-paired intervals, ADE/FDE on available labels,
original risk, dual passes, and support. Compare the learned initial
forecast to the old half-peak anchor as an unmatched initial-plan screen,
not a matched-budget advantage. Diagnose the single-query relaxed
utility bound for the new diagnostic initial plan before spending on a
cross-query selector. Moving A and independent confirmation stay sealed.
