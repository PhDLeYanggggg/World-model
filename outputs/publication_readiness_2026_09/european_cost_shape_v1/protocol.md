# Shape-Constrained Expected Harm Readout

## Material Passport

Source-development mechanism experiment, registered before fitting or opening
new held predictions. Parent cost-mass verification at ca5d6e31 binds all
controls. The parent found 99/432 easy-mass constraints infeasible at slope8;
even matched training mass did not transport as accurate held-scene costs.

fresh_run:864 small monotone readouts, fitting certificates and source-held
evaluation. cached_verified:1296 nested inner neural heads,432 outer heads,
causal features, forecasts and raw/origin-L2/moment controls. not_run:new
neural or forecaster training, new policy or independent selection, reserved
calibration, confirmation. No deployment change, Stage5C or SMC.

## Question and Fixed Methods

Can fixed shape freedom preserve fitting harm mass while reducing squared
cost error, and does that transport beyond origin scaling? No split, target,
input bank, primary metric, guard or risk tolerance changes.

Retain all144 views (six source assignments,three seeds,two forecast families,
four outer localities),cost-only,true cap auxiliary and shuffled auxiliary.
All fitting scores are honest nested OOF, with the outer locality and entire
forecast-producer roster excluded. Recheck row hashes, preprocessing lineage,
inner easy cuts and target hashes against the completed parent. Crossed
two_cut3 heads are not substituted for honest inner OOF scores.

For raw H_all and envelope E use r=H_all/E. For raw H_easy use
u=H_easy/H_all;zero denominators have fraction0. Inputs obey nested causal
bounds. Fit increasing piecewise linear functions g on fixed knots
[0,.1,.3,.6,1],with knot ordinates in[0,1]. First predict E*g_all(r).
Then predict corrected_H_all*g_easy(u). Denominator columns are unchanged.
Neither inference function accepts labels or future positions.

Each component minimizes equal-locality weighted squared error over known
positive-envelope fitting rows. Compare shape_L2 without a moment constraint
and shape_mass with weighted predicted mean equal to weighted target mean.
Both use the same fractions and knots. These are sequential component fits,
not a joint optimization of easy and all harm. Easy caps differ across modes.
Relative to the parent, both shape and the nested fraction parameterization
change; only the comparison between these two new modes isolates the added
mean constraint. No slope-bound expansion or penalty/threshold search.

Nonnegative knot increments plus one slack coefficient form a six-element
simplex. Enumerate all63 supports and solve their quadratic KKT systems.
Require feasible simplex/moment constraints and normalized convex first-order
gap <=1e-7, checked against every feasible vertex. A constant fraction makes
the mass problem feasible when nested target means obey the causal envelope.
The fit must fail visibly if feasibility or optimality checks fail; do not
silently replace the result. A zero capacity predicts zero. Synthetic tests
compare the solver against an independent SLSQP implementation. No knot count
or position is selected from held data.

Training moment equality is not conditional calibration, held-scene calibration,
conformal validity or physical safety. Shape fitting introduces no new neural
representation. A better training fit alone cannot justify another deployment.

## Evaluation and Screens

Freeze every coefficient and held prediction receipt in Git before scoring.
The current source localities have historical exposure and remain development.
Independent roles stay closed. Report all shape_mass vs raw/L2/mass/shape_L2
comparisons in all three arms;shape_L2 vs frozen L2 in all arms;true vs cost
and shuffled under shape_mass. Do not select a favorable assignment or arm.

Primary remains positive-envelope expected easy-harm MSE. Guards remain top10
harm capture,coverage-log error and all-envelope H_all MSE. Compare full-input
cost-only shape_mass to raw and frozen L2:both require six positive primary
intervals,no negative or missing guards,and all fitting mass constraints and
solver certificates supported. Auxiliary information separately requires true
to beat both cost-only and shuffled with six positive primary intervals and
no negative/missing guards. No coverage-only replacement for a failed primary.

Three seeds are averaged within each locality,then3000 paired bootstrap draws
of four localities per assignment,seed71429. Six overlapping assignments yield
descriptive unadjusted intervals,not six independent datasets. Full/motion
have different forecasts and event populations,not a matched feature ablation.
Report absolute MSE and mass/coverage alongside relative gains. Keep missing
support and failed cells visible. No ADE/FDE or easy-degradation claim follows
from an expected-cost readout.

## Runtime and Stop Rule

Use native arm64,CPU4/interop1,workers0. Pilot one full view before the complete
matrix. Store per-view receipts,PID/UTC heartbeat and resume exactly completed
views. Preserve at least10GiB free disk. CREATE observation is read-only;these
cached arrays and small quadratic fits do not justify a new GPU job.

If the shape experiment fails its scientific screens,stop cycling generic
global scalar/shape readouts. Next work must address causal context,rare-event
support or a concrete falsifiable transport mechanism;do not reopen independent
outcomes or repeat threshold searches. A failure remains part of the evidence.

Obs8/pred12 native annotation steps,detector pixels. No metric/seconds,
human-gold,true3D,foundation or physical-safety claim. Stage5C/SMC stay off.
