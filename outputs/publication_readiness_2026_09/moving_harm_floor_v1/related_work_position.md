# What This Experiment Can Contribute

Official primary-source abstracts and publication metadata rechecked on
5 November 2026. The initial pass was metadata-only; the method-section review
below followed while the fixed experiment trained. It is not proof verification.

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

## Method Review During Training

Mao et al., section4, equations3-4: with the predictor frozen, expert costs
weight the routing surrogate. This is a relevant matched control for routing;
it does not make our four-output maximum an instance of their proven loss.
[Official paper](https://raw.githubusercontent.com/mlresearch/v235/main/assets/mao24d/mao24d.pdf).

Cortes et al., sections3-5: input-expert rewards induce a cost-sensitive
classification problem. MILD uses expert-specific margins inside a
reward-weighted logistic objective. That differs from raising small harmful
target weights in our binary admission surrogate. Their expected-loss
consistency results do not establish our per-recording 2% contract or repair
missing reference support. This last comparison is an inference about the two
different objectives, not an extension of their theorem.
[Official paper](https://raw.githubusercontent.com/mlresearch/v306/main/assets/cortes26b/cortes26b.pdf).

Reviewed text: Mao PDF pages2,4-6 and Cortes pages2-8; rendered equation-bearing
pages5 and6 respectively were checked visually. Appendices/proofs and their
code were not verified. No new prior-work implementation or experimental
claim follows from this review; the running protocol was not changed.

Downloaded PDF SHA256:
- Mao: `06d3b0b82937a157b23084ba8ba6c7110420edec0ecd7e545f57fee18a9a1ed5`.
- Cortes: `e1e335bfa5de101a580197daf0bdbb4d6a0660c1a1852f4b6721fb095420ed72`.

Third-party PDFs and temporary renders are not included in the repository.
