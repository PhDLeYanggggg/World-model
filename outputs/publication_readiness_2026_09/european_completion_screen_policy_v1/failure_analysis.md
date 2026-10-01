# What the Completion Repair Does and Does Not Fix

## Failure Taxonomy

1. **Missing-outcome support was a real but partial blocker.** Geometric bounds
   recover nine of41 unknown-only candidate views without treating missing harm
   as zero. All149 prior known-outcome risk failures stay unsupported.
2. **Source selection still prefers a prior.** The three supported trained
   candidate views lose to initial priors on the unchanged known-validation
   utility ranking. All nine nonfallback source choices are stepzero. This
   prevents a learned-head or latent-contribution claim.
3. **Source support does not transport automatically.** Locality008's recovered
   prior policies exceed the selected-risk budget on locality020 in every seed.
   A bound on missing labels in one fixed sample is not a statistical certificate
   for a different locality, even if every future input is properly excluded.
4. **Coverage is not ranking.** Intervention increases0.264% to2.847%, while the
   same-count actions remain identical. The tiny primary gain cannot be credited
   to better discrimination among simultaneously eligible agents.
5. **Net ADE is not positive-harm control.** Whole-population easy ADE does not
   degrade, yet five nonempty views exceed the2% easy selected-risk budget.
   Reporting only easy net ADE would conceal the failure.

## Consequence for the Research Direction

The chain now separates three mechanisms with executed controls: global cost
MSE improves without decision improvement; validation utility screening mostly
abstains; accounting for missing outcomes restores some coverage but not learned
superiority or transferred conditional-risk control. Continuing to tune the
same veto or transfer threshold is not the next useful experiment.

Next test whether the present causal inputs support useful conditional cost
learning with a standard non-neural regressor under the **same European
source-only optimization/validation partitions**. Compare it against the frozen
neural cost heads and their priors, retaining the five benefit/harm/reference
targets and the2% screen. A matched comparator can distinguish a neural fitting
limitation from a failure shared by a flexible conventional predictor. It cannot
prove irreducibility if both fail.

Reuse the existing forest tooling where compatible, but do not relabel the old
SDD `forest_cost_v1` result as this control: its data, targets and weighting differ,
and its primary forest-versus-neural interval was inconclusive. Register the
new source/feature/target contract and run a memory/disk pilot before any full
fit. No hyperparameter sweep on transfer labels, no new locality exposure, no
claim that another estimator is itself a novel world model.

If the standard comparator also fails source-validation risk and utility, move
to a concrete history/label-quality or forecast-opportunity repair supported by
those fitting diagnostics, not another unsupported architecture claim. Independent
calibration, public strong comparisons under the same final protocol, and fresh
confirmation remain necessary before a submission-quality method claim.

## Remaining Limits

These are exposed development comparisons with repeated windows and reused
forecasters. Nominal locality bootstrap intervals do not correct adaptive
research selection. Detector-derived trajectories are silver, and incomplete
future masks are not full trajectory ground truth. No seconds/metric/physical
safety statement is licensed. Deployment remains unchanged; Stage5C and SMC off.
