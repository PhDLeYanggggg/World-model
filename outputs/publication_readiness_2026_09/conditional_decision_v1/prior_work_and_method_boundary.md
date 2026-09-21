# Conditional Cost Reliability: Prior Work and Method Boundary

Checked 2026-09-21. This focused primary-source review follows the completed
[cost-head fit diagnosis](../cost_head_fit_forensics_v1/conclusions.md), not a
new prediction experiment. It is not a systematic review or certification of
novelty. Searches and section-level inspection were performed with AI assistance;
no independent reviewer or human full-paper reading is claimed.

## Question

How should a baseline-relative forecaster learn costs that remain informative
where its policy actually intervenes, while respecting an explicitly defined
easy-case risk constraint and scene-level dependence?

This is narrower than asking whether a larger Transformer can reduce trajectory
loss. Our frozen heads already fit aggregate costs better than a constant, but
23 of 24 original eligibility groups lose on their own fitting data. Eligibility
is a score-defined group, not a scene-solver decision; those fitting outcomes are
not independent validation or a deployment test. Domain shift alone cannot
explain failure that is already present in-sample.

## Direct Prior Art

| Primary source and inspected scope | Established result and M3W boundary |
| --- | --- |
| Elmachtoub and Grigas, *Smart "Predict, then Optimize"*, Management Science 2022; [arXiv v5](https://arxiv.org/pdf/1710.08005v5), sections 2-4; [published metadata](https://doi.org/10.1287/mnsc.2020.3922). | SPO measures decision regret; SPO+ is an optimization-aware surrogate. The inspected setup has uncertain objective costs and a known feasible set. Our predicted-harm constraint also changes feasibility, so directly inserting our solver does not inherit its consistency result. Cost-aware learning itself is not novel. |
| Zhao et al., *Calibrating Predictions to Decisions*, NeurIPS 2021; [published paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/bbc92a647199b832ec90d7cf57074e9e-Paper.pdf), sections 2-4, especially Definition 2 and its cautionary paragraph. | Decision calibration concerns expected decision loss for specified decision rules. The inspected framework uses class probabilities; its rules depend on those probabilities rather than arbitrary additional features. It does not certify individual decisions. Our continuous costs, extra scene geometry and joint action space need a separate argument. |
| Hebert-Johnson et al., *Multicalibration*, ICML 2018; [paper](https://proceedings.mlr.press/v80/hebert-johnson18a/hebert-johnson18a.pdf), section 2 definitions and section 3, Algorithm 1/Theorem 2. | Calibration across computationally identifiable subpopulations is established, with support requirements. Its binary-outcome formulation is not an automatic guarantee for unbounded motion costs. Future-error-defined easy membership is unavailable at deployment, unlike a past-feature-defined audit group. |
| Mandi et al., *Decision-Focused Learning: Through the Lens of Learning to Rank*, ICML 2022; [paper](https://proceedings.mlr.press/v162/mandi22a/mandi22a.pdf), sections 3-4. | Pointwise, pairwise and listwise objectives over cached feasible solutions already connect prediction to decision quality. Ranking candidate actions alone does not establish the absolute harm accuracy required by our budget. A ranking loss is a useful comparison, not a new contribution by itself. |
| Mao et al., *Regression with Multi-Expert Deferral*, ICML 2024; [publication page](https://proceedings.mlr.press/v235/mao24d.html), [arXiv v1](https://arxiv.org/pdf/2403.19494v1), sections 2 and 4. | Fixed-predictor, label-dependent-cost regression deferral is direct prior art. Its bounded-loss theory does not automatically cover our unbounded error and structured risk constraint. The earlier matched deferral control remains required evidence, including its negative outcomes. |

The official Mao PDF endpoints failed in the browser parser (including the
linked publication asset's content type); the proceedings metadata and specified
author preprint were inspected. This is not a final-PDF/version-equivalence
claim. The four newly added bibliographic entries are in
[references.bib](references.bib); the existing Mao entry remains in
[deferral_references.bib](../joint_intervention/deferral_references.bib).

## What the New Diagnosis Does and Does Not Establish

1. **The failure is conditional.** Every frozen head beats constant-label MSE,
   yet every original eligible subset underestimates harm. Nine of twelve heads
   even overpredict harm globally. A global intercept correction is not an
   evidence-backed repair for that sign pattern.
2. **The desired guarantee differs from the old budget.** A population absolute
   harm cap does not imply relative easy preservation. The needed joint moment
   cannot in general be obtained by multiplying easy probability by mean harm.
3. **Squared loss is not intrinsically invalid.** With finite second moments,
   its unrestricted population optimum is the conditional mean. If that mean
   conditions on all information the action uses, expected selected costs are
   correct by the tower identity. Our finite fitted heads are not that oracle.
   The findings do not prove that decision-focused learning is necessary or
   that longer training could never help.
4. **A calibrated unary score can still fail after composition.** A solver uses
   other agents and geometry in addition to that score. Calibration only within
   score bins need not survive this selection. The audit must concern the full
   frozen policy or a sufficient family of policy-dependent moments.
5. **The current interaction claim remains unsupported.** At matched count and
   unary geometry, the completed frozen comparison gives 21 zero, two tiny
   favorable and one adverse ADE contrasts. Every control fails easy preservation.
   Addressing cost reliability is a justified next comparison; it does not
   establish that joint selection will become useful.

Items 2-4 are elementary mathematical observations, with derivations and exact
finite examples in [risk_identities_and_counterexamples.md](risk_identities_and_counterexamples.md).
They are not presented as new theorems or empirical improvements.

## Smallest Discriminating Follow-Up

The [primary-metric decision](../source_population_v1/metric_decision_pending.md)
is still pending. The following is a conditional comparison design, **not a new
registration, approved loss, fitted policy or claim of progress on test data**.

First freeze the authorized metric, scene weighting, reference, easy definition
and separate producer/head/selection/calibration/test roles. Producer OOF alone
is insufficient: a cost head fitted on all those OOF rows has no held-out head
assessment. Historical exposed recordings remain development data.

Then hold forecasts, feature availability, capacity and fitting budget fixed:

| Comparison | What it distinguishes | Required readout |
| --- | --- | --- |
| Existing cost MSE versus the same cost head with a dedicated, development-only conditional recalibration split | Whether score bias is repairable without changing candidate trajectories or the main fitting objective | Held-out cost bias within prespecified eligibility and actual intervention groups, coverage, paired loss and harm |
| Cost MSE versus a matched decision-ranking/deferral objective | Whether global fit is using capacity on errors irrelevant to the action boundary | Decision regret plus cost magnitude error; neither ranking accuracy nor MSE alone is sufficient |
| Unconditional harm target versus an explicitly easy-weighted joint-cost target | Whether risk-target alignment matters, apart from score calibration | The same authorized easy numerator and denominator, retained tail/missing-label accounting, and complete non-easy outcomes |
| Geometry-aware independent versus joint decisions, after freezing each cost readout | Whether multi-agent coupling adds anything beyond reliable unary decisions | Matched intervention counts/budgets, full observed membership, paired trajectory and proximity outcomes |

These are hypothesis-isolating controls, not a request to fit every combination
in a large sweep. The next registered experiment should first test the smallest
necessary pair, with a fixed stopping rule. If candidate headroom is absent under
the authorized metric, repairing a selector cannot create better trajectories.

Do not choose groups from final-test residuals, adapt thresholds on calibration
failures without revising the guarantee, or count overlapping windows as
independent scene support. Missing selected outcomes remain unknown. Recalibration
data used to fit a readout cannot also serve as untouched final certification.

## Contribution Decision

Do not claim first cost-aware selector, first ranking-based routing, first
conditional calibration, first joint optimizer, a new safety theorem, or a
JEPA/Transformer contribution inferred from architecture alone. The plausible
research distinction is **baseline-relative conditional risk in a composed
multi-agent forecasting policy**, but usefulness and novelty both remain open.

A compelling contribution would need a precisely specified policy/target,
matched prior-art controls, useful held-out decision-region reliability, and a
nontrivial multi-agent benefit at the same safety level. If those comparisons
fail, shrink the claim; do not rename standard calibration as a new world model.
The present diagnosis can support a methods-motivation or limitations section,
not the paper's positive main claim. No submission-readiness upgrade is made.

## Search and Verification Scope

Queries on 2026-09-21 included `decision calibration predict then optimize
conditional expected costs regression paper`, `decision focused learning
calibration regret cost prediction Smart Predict then Optimize paper`,
`multicalibration regression downstream decisions subgroup calibration paper`,
and exact-title follow-ups. Only original proceedings, publisher metadata and
author preprints support the claims above. Secondary search results were not
used as evidence. This finite search cannot rule out equivalent structured
methods or constitute an exhaustive novelty review.

Fresh activity: primary-source inspection and exact synthetic math checks.
Reused evidence: previously verified frozen fit/risk/coupling reports, without
new model inference. Not run: new training, cost recalibration, independent
confirmation, any comparison under an amended metric. Stage5C and SMC remain
disabled. Pixel/dataset-local units, raw annotation steps and offline supplied
annotations retain their existing limitations.
