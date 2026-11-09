# Moving: can the same budget work across queries?

Use the v155 frozen held forecasts and half-peak anchor for the same
24 recordings, nine development sources and three seeds. No new
training or choice on Moving A or DroneCrowd. Use the original
query-balanced paired lower gains versus both CV and OLS4, the same
three recording-level 2% checks, and required easy reference support.

Compare two matched budget ceilings: keep each query's original
edit count and sum of CV-relative peak displacements, or pool
those same counts and peaks over the recording. At most one
path per row. Do not require spending a budget just to make an
intervention. Both use the same hindsight search and exact evaluator.

The base menu contains the anchor, OLS4, OLS8, the earlier completion
spline, EqMotion and PCT. Use 0.10, 0.35, 0.70 and 1.00 CV interpolants
of the non-anchor paths. The expanded menu adds the OLS-only,
dual-risk and diagnostic v155 paths at those same fractions.
The menus are nested. Independently replay each passing witness,
including risk, support, counts, radii and paired utility. Label solver
timeouts without a certified exclusion or a passing witness unresolved.

Also maximize each reference's utility separately over arbitrary
directions and the same total peak mass. This relaxation drops risk,
integer edit counts and realizable trajectory shape; it is an upper
bound, not a policy. The finite-menu solver then separates the
effect of moving interventions across queries from adding predictors.
Report verified counts and uncertainty with a source-paired bootstrap
after seed averaging. Retain negative and unresolved findings.
