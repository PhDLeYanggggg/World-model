# Failure Analysis: Better Fitting Does Not Repair Source Transfer

## Controlled Change

The prior 32-epoch experiment left optimization and source transfer confounded.
This experiment continued all eight saved models, preserving optimizer/random
state and every setting other than the update budget. It added 32,871 primary
updates in complete epochs, with no held-out stopping or checkpoint selection.
All fits finished and reproduced before final development evaluation.

## Findings

1. **The optimizer can fit the donor bank.** Fitting loss improves in all eight
   folds. Source076 falls from 0.525897 to 0.000365 after 4,096 extra updates;
   source067 falls from 0.483918 to 0.004197. This is real learning of the fitted
   objective, not a stalled Torch runtime or unchanged parameters.
2. **Most source-held errors worsen.** Plain neural risk MSE improves in only
   two folds, source053 and source073. Source067 rises from 1.720374 to 6.344891;
   source076 remains poor, 7.738072 to 7.787639. More updates alone do not repair
   this fixed model's transfer. These observations are consistent with overfitting
   and source-conditional mismatch, not proof of a unique cause or irreducibility.
3. **Safe decision quality is still insufficient.** Continued transport changes
   decision cost relative to the short network by +2.053162%, -9.141560%,
   -1.113508% reduction across the three contexts. Against shared64 the reductions
   are -4.719181%, -68.647572%, -6.692049%. None reaches the registered screen.
4. **Harm is not hidden by lower loss.** Harmful admissions are 214/498/1219,
   versus 0/1/0 for shared64 and 94/353/1288 for the short transport. Known selected
   easy-risk ratios are still high in most sources. Those normalized, selected
   known-label ratios are descriptive, not the full-policy physical-risk estimator.
5. **Reference correction loses value as donor residuals shrink.** Context0
   plain versus transported MSE is nearly identical after longer fitting; its
   transport decision cost is worse than plain. The near-zero donor fitting
   residual cannot correct a source-held neural error merely by averaging it.
   This is an observed failure of this in-sample residual readout, not a proof
   that all retrieval or cross-fitted residual methods fail.
6. **Independent fitting-source coverage is narrow.** Verified metadata shows
   two folds train on one source component each and six on two each. Source076
   has 716 donor windows and 112 conservative identity buckets from two sources.
   Window counts are not independent scene counts. This limits what a risk head
   can learn about unseen-source tail harm; it does not authorize blind pooling.

## What This Does Not Show

No fold satisfies the registered last-quarter 1% fitting-plateau diagnostic.
Mathematical convergence is not established. However, the achieved large fitting
reductions together with worsening source-held error weaken an undertraining-only
explanation. Another unstructured epoch increase is not the next primary remedy.

The original 2% all/easy positive-harm, easy-net and support requirements remain.
Empty support/control is NA; there is no identity exemption. The component
screen passes 0/3 and the original 24-group primary remains NA, not passed.
These are already-exposed own-B development results, not a new independent test.
One seed and overlapping windows do not establish a main-paper confidence interval.

## Next Decisive Step

Audit the eligibility of additional already-authorized SOURCE forecast banks
under each frozen predictor's teacher-training ancestry and risk-target definition.
The immediate question is whether the current one/two-source fitting restriction
can be expanded without source or teacher leakage. If compatible donors exist,
predeclare one source-aware regularized/cross-fitted risk comparison with matched
data and the same original rules. If they do not, record the actual missing source
coverage and avoid pooling incompatible critics or opening reserved data.

No new trajectory model, deployed neural improvement, metric/seconds claim,
Stage5C or SMC is established. Deployment remains unchanged. This experiment
changes the next research action; it does not establish submission readiness.
