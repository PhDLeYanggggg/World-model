# What This Experiment Can Contribute

Official primary-source abstracts and publication metadata rechecked on
5 November 2026; this refresh is not a full rereading or theorem verification.

[Mao, Mohri and Zhong, ICML 2024](https://proceedings.mlr.press/v235/mao24d.html)
study regression with multiple experts, including fixed-predictor two-stage
deferral, bounded losses and instance/label-dependent costs. Therefore learning
when to use a frozen predictor is established prior work, not our novelty claim.

[Cortes, Mao, Mohri and Zhong, ICML 2026](https://proceedings.mlr.press/v306/cortes26b.html)
formulate cost-sensitive deferral under expert imbalance and introduce MILD.
Their stated experiments cover image classification and LLM routing. This
does not establish that our harm-floor loss is their algorithm or inherits
their guarantees; no matched MILD implementation is added in this round.

The present three-arm experiment is a mechanism test: does redistributing
positive-label penalty improve decisions beyond a uniform increase of equal
total coefficient mass? Even a positive result would not make a weighted loss
alone a new world-model contribution. The larger unsolved question remains
causal, scene-level neural intervention with baseline-relative harm control,
partial labels, support requirements and independent scene confirmation.

The formal paper still needs successful original-rule results, independent
supported evidence, the planned seeds and matched strong forecasting/deferral
controls. Neither a favorable loss curve nor a diagnostic target-safe oracle
replaces those requirements. See the broader [related-work record](../cost_decision_related_work_20261029.md).
