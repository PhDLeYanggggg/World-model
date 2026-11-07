# From Candidate Availability to Observed Intervention

## Controlled Setup

We keep the forecast bank, direct policy, and original 2% recording risk
and reference-support rules fixed on exposed Moving A. Each of 42 cells
uses seven fitting sources, six calibration sources and one held source.
The 102 views repeat 34 recordings over three forecaster seeds. The task
uses eight observed and twelve forecast annotation steps in image coordinates.

### Table 1. Positive supported actions in the first two eligible queries

| Available actions or final choice | Zero-direct views hit |
| --- | ---: |
| Whole bank | 15 / 18 |
| Flat previous shortlist | 2 / 18 |
| Family-diverse shortlist | 6 / 18 |
| Gain-only winner | 0 / 18 |
| Risk-penalized winner | 0 / 18 |
| Event-score winner | 1 / 18 |

These are descriptive counts over the 18 original zero-direct views.
A hit requires a strictly positive ADE gain and a standalone pass under
the unchanged observed rule. A shortlist hit still needs a decision
rule to choose the action. The event-score winner is a posthoc readout
of the same fitted model, not an additional calibrated policy.

## Choice At Fixed Coverage

The fixed shortlist puts the highest original-score candidate from
each family first, then fills the remaining slots up to eight. A ridge
model trained on donor-excluded fitting shortlists estimates utility,
risk costs and the useful-action event. The gain and risk choices see
the same shortlist and predicted outputs. In the unthresholded
first-query control, both make 6,998
switches at identical queries. Risk minus gain changes paired
completion-lower utility by +0.00156 percentage points
(95% source interval [-0.00688, +0.01348]). Both have
5/102 complete passes in this diagnostic.
With donor-only threshold selection, the gain and risk arms have
2 and 2 complete passes,
versus 2 for direct. These do not support
improved calibrated intervention.

## Relation To Deferral

[Mao et al. (ICML 2024)](https://proceedings.mlr.press/v235/mao24d.html)
already study multi-expert regression deferral with fixed predictors.
Here we isolate candidate availability, an eight-slot shortlist, and
final choice under the original trajectory-risk contract. We do not
implement their surrogate or claim their consistency guarantees.
The gap between a useful action being available and being chosen
remains the central empirical limitation. No independent confirmation
or deployment follows from these results.
