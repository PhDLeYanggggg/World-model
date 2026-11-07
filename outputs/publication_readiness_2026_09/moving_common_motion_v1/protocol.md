# Bounded Common Motion With Fixed Relative Geometry

Use the exposed 34 recordings, 14 source groups, three forecaster
seeds and the native 8/12 schedule. Separate the mean displacement
path at a query from per-agent relative motion. Keep every relative
vector of OLS4 unchanged. Replace only its mean with a convex
mixture of the OLS4, CV and frozen self-kinematic means. No camera
motion or calibrated world coordinates are inferred.

Compare plain OLS4, CV-centred OLS4, kinematic-centred OLS4, their
uniform mixture, the donor-selected fixed mixture, and a learned
query mixer. CV is the original risk reference. The 24-wide causal
mixer sees past motion and candidate-centre offsets. Fit for 480
updates in batches of 12 queries, balance sources, and select the
checkpoint among 160, 320 and 480 only on donor lower utility.
The training loss is query-balanced available per-row ADE plus the
actual translation radius on wholly unknown rows. Fit on seven
sources, select on six, evaluate the remaining source. Do not use
the held outcomes to choose a mixture or fit preprocessing.

Primary: learned vs OLS4 paired completion-lower utility with
the same CV denominator. Also compare with CV-centred and donor
fixed controls and retain the original 2%/support readout versus CV.
The full-scene forecasts are not matched-count policies; report
intervention counts as outcomes, and make no deployment change.
Use source bootstrap; require identical OLS4 relative vectors for
every query and every mixture, to numerical tolerance.

These are development data. No external roles or outcomes opened.
