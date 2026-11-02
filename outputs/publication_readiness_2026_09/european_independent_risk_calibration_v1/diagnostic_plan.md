# Descriptive Failure Classification

This plan is fixed before any calibration target construction or scoring.
It does not modify model fitting, lambda selection, policy decisions, support,
the original 2% tolerances, the cohort or the final-confirmation boundary.

After the complete fixed-grid readout, classify each of the 918 repeated
pipeline-record views at every multiplier and policy. Record overlapping
flags for unavailable reference denominators, observed positive-harm excess,
signed easy-net excess, completion-upper excess, empty intervention and
selected wholly unknown outcomes. For a disjoint table use this priority:

1. Unsupported reference denominator.
2. Observed positive-harm excess.
3. Easy-net-only excess.
4. Wholly unknown-row completion-bound-only excess.
5. Supported under the original recording checks.

Category counts must reconcile to the original supported-view counts. Keep
all recordings and all nine pipelines. Verify identical label rosters across
all policies/producers before reporting unique coverage. Do not count repeated
pipeline/overlapping-window observations as independent labels. A conservative
bound failure is not a claim that missing trajectories attain its bound.

This is a reporting diagnostic, not another threshold/model-selection trial
or an inferentially independent efficacy test. It cannot promote a policy or
change a failed gate. No final-confirmation or DroneCrowd data is accessed.

Also separate complete known-easy reference unavailability from merely empty
selected reference. The former is invariant to policy/multiplier: when the
whole known-easy floor-reference mass is zero, no selected subset can provide
positive known-easy reference under the same floor, labels and easy definition.
Report its repeated-view count, reconcile that invariance, and retain all
original requirements. This check distinguishes a structural support obstruction
from a critic or threshold error; it does not redefine zero support as safe.

Report descriptive paired contrasts of matched-joint, free-joint and uniform
policies versus independent for every fixed grid point. The known numerator
comes from matched per-locality utility totals. For wholly unknown rows,
compute the actual candidate-versus-control path disagreement; never subtract
two floor-relative lower bounds. Use the same reference-weighted locality
totals and 3,000 equal-locality bootstrap replicates. These contrasts do not
choose a model, seed, multiplier or a new primary test, and do not undo fitting
the calibration parameter. Partial missing steps remain unaddressed.
