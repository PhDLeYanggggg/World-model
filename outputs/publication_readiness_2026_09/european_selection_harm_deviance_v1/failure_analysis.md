# Failure Analysis: Better Cost Sensitivity, Worse Policy Utility

## The Registered Result Failed

Dual H/EH deviance versus the original mean_absolute control has FULL paired
completion-lower effect -1.9774pp [95%CI:-2.6565,-1.2590]. The easy-only control
also loses: -1.9449pp [-2.6086,-1.2496]. Neither is deployed. A positive mean
over the physical floor does not override this negative primary comparison or
the failed original risk contract. Both arms and all recordings are retained.

## Not A Runtime Or Control-Reproduction Failure

All18new Torch fits completed the original budget, with every SOURCE query
visited at least four times. Initialization, model size, normalizers, sampler,
draws and budgets match the original control. The original100step and complete
control states reproduce exactly. Both first-producer new full refits match.
The actually disk-interrupted2000->8192model also matches a direct full fit in
weights, optimizer, sampler, trace and RNG. All18cost inferences and54development
metric groups replay exactly. No speed-based reduction was made.

The disk guard was real, but its root cause remains unconfirmed. Free space
recovered without deleting data or changing the guard. Model optimization was
resumed, not declared complete from a partial checkpoint. The original failed
run log is preserved. None of these engineering checks establishes efficacy.

## What Changed On SOURCE

The pooled selected easy-harm prediction/observed ratio improves from0.00364
for mean_absolute to0.2051 for easy-only and0.2140 for dual. Dual ordinary harm
ratio improves from0.2811 to0.4733. Nevertheless, its known selected SOURCE
all/easy risks remain3.69%/5.57%, above2%. These are fitting-data ratios on
different selected pools, not paired errors on fixed selected actions or
held-out calibration. Selection and regression both change.

The common all-legal bank provides a fixed-set check: all arms have5,267,952
repeated observed action views and matching target masses. Dual H normalized
MSE falls from0.00347079 to0.00315142, but EH MSE increases from0.000415010
to0.000486812. The easy-only EH MSE is0.000493311. Thus selected-pool ratio
improvement is not a blanket improvement in fixed-bank regression accuracy.
These are SOURCE resubstitution errors, not downstream validation or estimates
of a conditional mean on new scenes. The exact pooled masses and all five cost
errors are recorded in fixed_bank_cost_diagnosis.json.

This supports a narrow optimization effect, not a reliable cost model. The
new objective strengthens underestimated-cost gradients in a synthetic check;
that does not prove floating-point underflow caused the real failures. The
older stable easy-deviance implementation was already numerically stable and
its negative144fit result is not rewritten.

## Why Lower Easy Net Is Not Deployment Success

On the complete exposed development cohort, known ADE gain over the floor
falls from1.5037% to0.08694% for dual. Worst signed easy net falls from8.4955%
to1.6223%, but only30/252repeated recording views pass the complete original
contract. There are134known all-risk violations,133known easy-risk violations
and64undefined selected-easy views; these categories overlap. Benefits cannot
cancel the positive-harm numerator, and empty reference is not a measured safe0.
Thus failure remains even apart from unsupported denominators.

## Policy-Region Diagnosis

An explicitly post-readout label-free analysis reads frozen cost predictions
and choice arrays, not future targets. Across342,918repeated pipeline-target
views, the original policy codes82,848interventions; dual codes7,500. For each
originally selected action under the new costs:

|Exclusive reason|Repeated choices|
|---|---:|
|New guard rejects, new gain remains positive|62,198|
|Both new gain and new guard reject|16,398|
|New gain alone rejects|673|
|Neither rejects|3,579|

Guard-only rejection accounts for about75%of original interventions.77,182old
interventions now fall back. This is primarily broad guard rejection within
the original choice region, not merely a changed argmax ranking. Keeping old
utility alone cannot restore those rejected original actions under the same
new guards. This is not a bound over every alternative bank action, a measure
of physical harm, or proof all rejected actions were beneficial. Choice codes
can also differ while forecast paths coincide.

## Hypotheses And Next Decisive Experiment

The result contradicts the simple hypothesis that fixing small-cost gradients
alone yields a stronger safe policy on this framework. It does not establish
that all cost-divergence objectives fail. Severe fitting-side conditional-cost
error remains, while more conservative prediction removes useful interventions.
Representation insufficiency, finite head capacity, rare-event support and
unpredictable future behavior are not separated by this experiment.

The next priority is not another scalar threshold sweep on exposed calibration.
First inspect existing strong-regressor baselines for this exact bank, schema
and split. If absent, run a fixed, SOURCE-matched nonlinear cost-regression
capacity control and compare fitting versus exposed-development conditional
cost error. That can distinguish a weak neural head from weak causal information
before another architecture/loss search. Historical forest or hurdle results
on other banks/teacher scopes must not be relabeled as this matched experiment.
No such new capacity experiment is claimed complete here.

## Remaining Paper And World-Model Gap

The divergence is classical; adopting it is not a method contribution. This
experiment supplies controlled negative evidence and a reproducible recovery
path, not submission readiness. Strong public-method comparisons, a useful
joint policy under the original risk requirements, complete anonymous model
reproduction and independent final confirmation remain incomplete. No new
JEPA, Transformer trajectory or world-dynamics contribution is demonstrated.

The task remains EuropeanSquares detector/tracker silver, obs8/pred12/raw
stride12 and local image pixels. Available-label ADE is not full12 ADE; the
unknown-row bounds omit missing steps inside partial trajectories. No metric,
seconds, human-gold, physical-safety, true3D or foundation claim is made.
Calibration private arrays, confirmation and DroneCrowd stay closed in this
experiment. The older calibration cohort remains disclosed as exposed in its
own experiment. Stage5C and SMC remain off. Deployment does not change.
