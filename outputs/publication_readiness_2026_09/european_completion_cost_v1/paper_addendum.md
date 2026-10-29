# Manuscript Evidence Addendum: Missing-Outcome Supervision Is Not Enough

## Research Question

Does adding valid finite-completion endpoints to cost-head supervision repair
the gap between observed-label training and complete-policy risk accounting?

## Controlled Design

We train 72 cost heads: six source contexts, three controller seeds, two fixed
candidate banks and two supervision modes. The trajectory producer is fixed.
Both modes use the same source-balanced causal query stream, initialization,
normalization, 2,000 optimizer updates and five-cost asymmetric objective.
Known-only training masks missing-outcome losses; completion-aware training
adds separate conservative lower/upper endpoint targets. Original evaluation
labels are unchanged. These targets are not imputed observations.

The study is restricted to source fitting data. All endpoints are frozen
before readout. Every planned group and undefined estimate is retained, with
3,000 locality resamples after averaging context/head-seed views. These
intervals characterize fitting contrasts, not independent generalization or
correction for repeated research selection.

## Result Suitable For An Honest Negative-Result Section

Completion-aware training reduces mixed-bank known ADE gain from 0.581% to
-18.512%, a paired change of -19.093pp (95% fitting CI [-50.820, -1.022]).
Required recording-risk support falls from 77 to 26/444. The count-matched
completion-lower contrast is -0.351pp (CI [-1.188, 0.349]); count matching is
only a utility control and does not certify safety. Separate primary and
fixed-region replays are exact.

On the old policy's fixed selected region, the new estimator improves the
median easy-harm prediction ratio from 0.0199 to 1.6523. On its own new selected
region, the ratio is 0.000628, despite an all-legal ratio of 4.7105. These
conditional summaries exclude zero denominators only from the ratio median,
report those undefined counts explicitly and never substitute for the full
risk gate. The result illustrates why an apparently conservative target and
better fit on one selection region do not certify a newly induced policy.

## Restricted Support Observation

For a fixed floor/easy event, selected easy-reference mass cannot increase
under row deletion. Since the known-only policy lacks that support in 62/444
views, any nested veto policy still lacks it there. This elementary argument
limits the fixed-head repair class; it is not a novel general impossibility
theorem, a population guarantee or a claim that all M3W variants are infeasible.

## Positioning

Learned regression deferral and missing-annotation deferral already exist;
neither routing regressors nor handling missing labels is claimed as a first.
See the [primary-source note](related_work_note.md). Our current evidence is
an empirical limitation of this particular cost learner and completion rule.
It does not yet establish a new competitive method or an A-conference claim.

## Claims Not Licensed By This Addendum

No independent improvement, selected-risk guarantee, successful world dynamics,
missing-at-random identification, metric/seconds prediction, human-gold labels,
true 3D, foundation model, Stage5C execution or SMC. The candidate policy is not
promoted. Negative controls remain in the result ledger.
