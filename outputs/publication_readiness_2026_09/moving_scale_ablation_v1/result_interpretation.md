# Interpretation: Local Utility, No Safe Repair

## What Was Tested

Six real neural continuations remove one explicit absolute-scale input. The
original2%risk/easy-net/support contract, decision thresholds, labels, proposals
and sampling stay unchanged. This is a matched feature ablation of risk heads,
not a new trajectory dynamics model or full coordinate-invariant pipeline.

The prior causal audit motivates the test:57.05%of proposed source078rows have
log-scale outside their own fitting range. Association alone did not establish
cause; this retrained comparison supplies a limited intervention on the model.

## Positive And Negative Results

Main MSE known ADE gain over the floor improves3.6006%to4.1531%, hard gain
4.6257%to5.4549%. The paired known change is+0.5525pp, missing-label completion
bounds[+0.0944,+1.0106]pp. These bounds are NOT statistical confidence intervals.
The model remains below the affine comparator's4.3233%known ADE gain.

The aggregate improvement does not repair safety. Worst defined easy-net
degradation worsens38.9046%to47.5913%. Full risk/support remains0/17recording
views. Known all-risk violations fall14to13, easy violations remain13, and
undefined selected-easy views remain2. The main prospective expansion screen
fails: the third context loses utility and the aggregate completion lower
contrast against affine isnegative. Original24group primary remainsNA.

Secondary signed-cost gain improves2.6928%to3.0085%, paired+0.3158pp with
completionbounds[+0.1141,+0.5174]pp. Worst easy-net worsens26.2254%to30.2521%;
support remains1/17, undefined selected-easy increases1to2. Its expansion
screen also fails. It does not become the primary result after inspection.

## Where The Tradeoff Occurs

- Source078, MSE: gain11.0330%to13.7144%, switch34.1817%to48.1067%. Its paired
  completion lower is+1.6891pp. There is a local utility effect of this ablation.
- Source078, signed-cost: gain0.5976%to0.5281%, not a rescue. Removing the scale
  input does not explain or reverse the previous surrogate's large loss here.
- Source066, MSE:131selectedrows become1, gain1.9381%to0.0080%. This loses useful
  interventions; the remaining selected-easy reference is undefined, not passed.
- Source077, recording46: selections1110to1238of3193rows. Known easy-net
  degradation38.90%to47.59%; unknown selected outcomes17to28. This is a concrete
  easy-harm failure, not merely the empty-support accounting problem.
- All seven sources are retained inper_source.md. MSE improves four and worsens
  three. No favorable-source restriction or posthoc threshold follows this readout.

## Fitting Is Not Uniformly Better

Complete empirical query-weighted MSE changes+48.83%,-3.44%,+1.29%across the
three contexts. Easy cost-weighted sign regret changes+15.51%,+19.20%,-4.00%.
In the second context, removing scale slightly lowers MSE yet raises false-safe
easy outputs73to81. In the third it raises false-safe easy outputs952to1024
while reducing their weighted sign regret slightly. Error count, cost and held
utility are different quantities. More beneficial interventions can coexist
with worse easy harm; an aggregate utility gain is not a risk guarantee.

## Failure Taxonomy

1. Explicit-scale dependence: partially implicated in source-specific utility,
   not established as the sole or dominant safety cause.
2. Harm discrimination: the model still admits known easy-harming candidates.
   Removing an input has not learned a reliable conditional harm boundary.
3. Source heterogeneity: useful interventions increase in some scenes and
   disappear in another. Only2/3/3sourcegroups train the respective critics.
4. Missing-reference support: unchanged and separate. Model fitting cannot
   create missing denominators; zero intervention is not recertified as passing.
5. Evidence scope: one producerseed and repeatedly exposed development sources.
   This is not independent cross-domain confirmation or a submission-ready result.

## Next Model Question

Do the shared all-risk/easy-risk representations trade off the two tasks, or
does the easy-harm head fail even on its own fitting support? Inspect the
per-task fitting gradients and realized error mass before registering a
decoupled-head comparison. Existing related gradient studies must be checked
to avoid repeating them. This next experiment isnot_run; no efficacy claim.
Do not add another threshold sweep, remove difficult sources, or relax2%.

Deployment unchanged. Prior failures remain. Reserved20Moving,DUT,DroneCrowd,H
remain closed. Nativeobs8/pred12imagepixels, not verifiedseconds, metric, true3D,
foundation or human-gold. Stage5C/SMCoff. The long-term research goal isunachieved.
