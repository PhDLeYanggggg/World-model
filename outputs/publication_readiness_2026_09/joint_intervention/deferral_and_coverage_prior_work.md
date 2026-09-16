# Deferral and Coverage: Prior-Work Boundary

Checked 2026-09-16. Focused primary-source review, not a systematic review or proof of novelty. Search terms: `learning to defer regression multiple experts cost sensitive regression surrogate consistency paper`; `selective regression baseline improvement risk control prediction defer expert paper`. Followed original proceedings and paper text, not third-party summaries. Source synthesis was performed inline with AI assistance; no independent expert review or human full-paper reading attestation is claimed.

## Directly Relevant Sources

| Source and inspected scope | What is already established | Consequence for M3W |
| --- | --- | --- |
| Mozannar and Sontag, ICML 2020, [proceedings abstract](https://proceedings.mlr.press/v119/mozannar20b.html) | Expert deferral is reduced to cost-sensitive learning with a consistent surrogate. | Learning when another predictor is preferable is not itself a new problem. This classification work is not a trajectory-regression reproduction. |
| Mao, Mohri and Zhong, ICML 2024, [published metadata](https://proceedings.mlr.press/v235/mao24d.html), [arXiv v1 sections 2, 4 and 5](https://arxiv.org/pdf/2403.19494) | Regression deferral supports fixed predictors, multiple experts, bounded regression losses and label-dependent costs. | A pretrained motion predictor plus a learned cost-aware gate overlaps directly with existing two-stage deferral. Do not transfer the paper's consistency statements to unbounded trajectory error or our structured constrained objective without a derivation. |
| Shah et al., ICML 2022, [sections 2.2 and 3.1](https://proceedings.mlr.press/v162/shah22a/shah22a.pdf) | Selective regression evaluates error against coverage; decreasing coverage can worsen subgroup performance. | Aggregate gain or fewer interventions cannot establish easy-case preservation. Their protected demographic groups are not our future-error-defined easy slice. |

The ICML 2024 publication PDF endpoint did not parse in the web reader; the original arXiv v1 full text was used for section-level verification, with the proceedings page verifying publication metadata. No claim of a version-by-version final-PDF audit is made. Bibliographic entries are in [deferral_references.bib](deferral_references.bib).

## Method Distinctions That Still Need Evidence

Our derived framing: with a fixed baseline B and neural forecast N, deciding between them can be expressed as comparing conditional losses. Subtracting the baseline loss gives a relative objective without making the routing problem new. Separating positive benefit and positive harm is useful for modeling asymmetric damage, but this algebra alone is not a methodological contribution.

The remaining hypothesis concerns *scene-level composition*: can a frozen unary risk estimate be used to select a better mixed multi-agent forecast when interventions interact, at comparable coverage? The existing JFP and joint-metrics review already prevents novelty claims for pairwise forecast compatibility alone. The new exact-count control removes one identifiable alternative explanation: joint policies might appear safer simply because they intervene less. It does not remove model-capacity, score-quality, subgroup-composition or selection-bias explanations.

Minimum comparisons before claiming the hypothesis is supported:

1. Identical forecasts and observed agent membership for independent and joint decisions.
2. Ordinary same-budget policies plus the new reference-conditional exact-count control. Report prediction error, positive harm, proximity proxy and both overall and labeled-subset intervention rates.
3. A cost-sensitive regression-deferral control using the same training-fold predictions, with the exact adapted objective disclosed. Following this literature pass, a [two-action control](../deferral_control/method_and_limits.md) has been implemented and verified on synthetic fixtures. Real fitting/comparison remains **not_run**; no theorem or published benchmark reproduction is claimed.
4. Explicit negative/zero-count and solver-failure accounting; no exclusion based on realized improvement.
5. Unexposed confirmation, approved physical-scene grouping and paired uncertainty. None is supplied by a literature citation or a successful optimizer test.

## Claim Decision

Do not claim first cost-aware selector, first regression deferral, first joint forecast optimizer, formal physical safety or a new risk theorem. The current proposed contribution is a testable combination of baseline-relative supervision and structured intervention under a dependency-aware evaluation protocol. Whether that combination earns a substantive methodological contribution remains an empirical and theoretical question. No new predictor comparison was run in this review.
