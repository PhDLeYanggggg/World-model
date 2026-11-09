# Smooth Lead-Dependent Proposal Mixtures

Forecast replacement can fail because the candidate path is inadequate,
even with a useful recipient. Test a shape change with the same analytic
experts, features, network dimensions, objectives, seeds, and sample draws.
The two networks predict three simplex vectors. The control uses their
arithmetic mean across all 12 steps; the treatment interpolates them
with quadratic Bernstein weights on the 12-step grid. Both always stay
inside the per-step analytic convex hull. Neither is a physical safety
guarantee. No labels, future masks, or neural teacher forecasts enter
proposal inputs.

Train inside the same nine disjoint three-source training, calibration
and held cells; 1,000 updates per model. The common objective is the
missing-step upper error difference relative to OLS4 plus twice the
positive parts of the CV-relative all/easy excesses, scaled by past
motion. These are local training surrogates, not a new evaluation rule.
For deployment eligibility select one CV blend amplitude per arm on
analytic-only calibration: 0.05, 0.15, 0.35 or 1. Require every recording
to pass the original rule and positive conservative source-macro gain
over CV and OLS4, or use CV. The calibration recipient rule is the
previous fixed analytic risk/preference selector.

Hold the 2,317 causal recipients fixed on held fitting recordings, with
their original frozen forecasts as a comparator. Compare the raw
new proposals and a shape-only control in which both new paths use the
smaller of their two past-computable peak changes on each selected row.
Freeze the learned predictions and calibration decisions before any
held scoring. Report full-grid paired gain, observed and conservative
neighbor error, the original 2% checks, and 3,000 source bootstrap draws
after seed averaging. This is exploratory on reused development roles;
do not open A or independent confirmation, change deployment, or
change the metric units, risk rules or 15/30 design.
