# Can OLS-anchored paths improve capacity at matched counts?

The previous gain models could not repair a joint policy inside its fixed
predicted budgets. This diagnostic separates a path-bank limitation from
the effect of those budgets. I use the same nine exposed source groups, 24
recordings, three seeds, eight observed positions and twelve forecast steps.
I retain every query's intervention count from the low-count joint policy.
I apply the unchanged observed recording-level 2% rule and do not impose
the previous head's predicted constraints. The two arms use identical
counts, outcome masks, references, missing-step debits, and solver limits.

The control uses the original analytic rays and five edit amplitudes. The
nested bank adds nine forecasts. Four gradually shift the latest position
toward a past-only OLS4 or OLS8 fitted intercept. Four add bounded
longitudinal or lateral trend to OLS4 using two disjoint past velocity
blocks. The last combines reanchoring and half of each trend. New paths
are supplied directly, not with extra interpolation amplitudes.

I solve the retrospective maximin gain over CV and OLS4 under the
original risk constraints. The LP and mixed-integer solve each have a
three-second limit. I replay candidate witnesses with the original
scorer on the full recording, and never interpret a timeout as a negative
certificate. If a nested solve times out, the old witness stays feasible.
The outcome-aware choices and capacities are diagnostic only, not a
trained policy. Report view counts, paired source-group bootstrap for
witness discovery, solver uncertainty and the path families used.
No DroneCrowd calibration or confirmation labels are loaded.
