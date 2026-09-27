# Method Positioning and Claim Boundary

## What This Experiment Can Establish

The intervention policy changes which agents receive a neural prediction. A
head with low average per-agent or whole-query error can still underestimate
the harm on the particular selected set. This experiment asks whether fixed
causal subset supervision, with an individual-error anchor, changes that
ordering under held-source development evaluation. It does not introduce a
new trajectory predictor or demonstrate new world dynamics.

The matched contrast holds representation, initialization, sampler, optimizer,
training budget, utility and query action count fixed. The auxiliary loss is
the only difference between the two new arms. Comparisons to the previous
pointwise control additionally change subset weighting; they cannot isolate
aggregation alone. Joint-policy comparisons can change intervention count and
must not be described as rate matched.

## Related Method, Not a Novelty Claim

Mandi, Bucarey, Mulamba and Guns, *Decision-Focused Learning: Through the Lens
of Learning to Rank*, ICML2022, train through objective values over subsets of
feasible solutions. Their solution-value pointwise loss squares predicted
versus actual objective differences. Our sum/mean error on fixed causal action
subsets is closely related; aggregation by itself is not a new contribution.
This test uses a shared individual anchor, frozen causal subset proxies and
source-separated neural-intervention scores. It does not reproduce their full
ranking/solver methods or provide a calibration/generalization theorem.

Primary source inspected: paper introduction and sections3-4.1, including the
solution-value loss; https://proceedings.mlr.press/v162/mandi22a.html and the
linked author paper. Broader public-method comparisons remain unfinished.

## Remaining Scientific Risks

- A frozen controller subset is a proxy, not the new model's actual selected
  set. Its fit sources exclude risk-fitting and held sources but share the
  experiment's already-opened development pool.
- Three predetermined subsets do not control every possible selection. Low
  average loss is not a certificate on the optimizer's eventual choices.
- Query means permit individual errors to cancel. The shared individual anchor
  reduces this degeneracy but cannot guarantee generalization.
- The signed score bases do not identify separate expected harm and error
  moments. Better signed ordering does not prove probability calibration.
- A fixed total-reference denominator makes abstention measurable, but cannot
  turn an undefined selected-reference ratio into a passing2%risk result.
- Twelve development localities with repeated views do not become hundreds of
  independent domains. Three seeds and locality bootstrap quantify only this
  development evidence. Independent confirmation remains closed.

No model promotion, formal risk guarantee or submission-readiness claim follows
from completing this experiment. The original raw-frame image-local detector
silver restrictions, no metric/seconds claims, and Stage5C/SMC prohibitions stay.

These percentages are not directly comparable to historical Stage37 t+50
figures: the data, observation/prediction protocol and protected reference differ.
Earlier exposed/contaminated results remain exploratory rather than recertified.
