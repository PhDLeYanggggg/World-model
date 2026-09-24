# Direct Easy-Weighted Harm Estimation Does Not Resolve Deferral Conservatism

This is a development-study addendum, not a submission-ready independent test.
It reports the registered easy_moment_v1 experiment with frozen forecasting and
cost-scoring producers. All four physical sites have influenced method design.

Let b denote baseline error, h the positive excess error of a candidate, d its
causal forecast disagreement, C the source-only easy cutoff, and e=1{b<=C}.
We fit E[eh/d|X], E[eb/C|X], E[e|X] and E[h/d|X] with one multi-output forest.
The direct policy requires d E[eh/d|X] <= 0.02 C E[eb/C|X]; a matched control
substitutes d E[e|X] E[h/d|X] on the left. Both also require the same frozen
positive predicted net gain and nonzero recent motion. Identical forecasts have
zero disagreement and never trigger intervention. These estimated inequalities
are not calibrated risk guarantees.

The matrix contains 36 fresh forests: four source exclusions, three seeds and
three fixed forecasting alternatives. Every forest uses the corresponding old
neural scorer's 768,000 source draw counts and reaches 128 trees. There is no
threshold or outer-outcome model selection. Evaluation preserves 175,756
past-eligible windows and reports incomplete/missing outcomes rather than
filtering query eligibility by future visibility.

The direct rule yields ADE gains of 0.1241%, 0.0134% and 0.0013% for damping,
Transformer and EqMotion, with no observed worst-site/seed easy degradation.
It fails to improve the utility/safety tradeoff of the strict protected neural
controls: Transformer retains 2.4368% gain with 1.0669% easy degradation under
the old rule. The new direct rule admits only 0.2321% of Transformer queries and
0.0099% of EqMotion queries. Its hard gains are effectively zero.

Direct-minus-product ADE differences are -0.08777, -0.01315 and -0.03768
percentage points. Nominal paired-site CI95 intervals are entirely negative.
Matching intervention counts does not reverse these comparisons. The result
rejects the proposed fixed decision rule, not the mathematical distinction
between a joint conditional moment and a product of marginals. A sufficient
pointwise positive-harm bound can sacrifice almost all useful interventions;
correcting target algebra is not equivalent to learning or calibrating an
effective population-level decision policy.

Only four development-exposed sites support the 3,000-resample bootstrap. Three
optimization seeds do not add independent sites. Complete-label fitting,
dependent windows and uncertain outcomes further restrict inference. All
decisions and aggregate reductions replay; a separate implementation verifies
errors, support, budget matching and paired contrasts. This establishes
computational reproducibility, not external validity. No new deployment,
metric/seconds claim, true-3D/foundation claim, Stage5C or SMC follows.

Full fixed controls, all site/seed results, losses, rejected-intervention counts
and source hashes accompany [the experiment report](conclusions.md). The next
method comparison must separate pointwise conservatism from aggregate budget
allocation without turning these exposed outcomes into a calibration set.
