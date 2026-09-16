# Forecasting Guarantees and the Remaining Method Claim

2026-09-16. `fresh_run`: focused original-source reading and analytical checks, not model training or empirical calibration. This is an AI-assisted, non-exhaustive research audit, not an independent expert review or a novelty certificate. Existing dataset/support findings are reused with their original provenance. No private data were uploaded. No scientific protocol, data role, risk tolerance or deployment policy is approved by this document.

## Original Sources Read

| Work | Text checked | Reading scope |
| --- | --- | --- |
| Lee, Barber and Willett, *Distribution-free inference with hierarchical data* | [arXiv v4, 2025-08-02](https://arxiv.org/html/2306.06342v4), first submitted 2023 | Definition 1, section 2.2 equation (6), Theorem 1; v3 was initially opened and superseded here by v4 |
| Mallick, Tchetgen Tchetgen, Dobriban and Lee, *Generalized Hierarchical Conformal Prediction* | [arXiv v1, 2026-08-16](https://arxiv.org/html/2608.15500v1) | Assumptions A1-A3, section 2.1, Theorem 2.1, appendix A.2; preprint, no peer-review status inferred |
| Zhou, Lindemann and Sesia, *Conformalized Adaptive Forecasting of Heterogeneous Trajectories* (CAFHT) | [ICML 2024 metadata](https://proceedings.mlr.press/v235/zhou24l.html); [arXiv v2 full text](https://arxiv.org/html/2402.09623v2) | Sections 2.1, 3.3-3.4, Theorem 1; publisher PDF retrieval failed, so full-text reading used the author preprint |
| Dixit et al., *Adaptive Conformal Prediction for Motion Planning among Dynamic Agents* | [L4DC 2023 original PDF](https://proceedings.mlr.press/v211/dixit23a/dixit23a.pdf) | Sections 3-4, Corollary 3, Theorem 7 |

Bibliographic entries are in [uncertainty_references.bib](uncertainty_references.bib). This supplements, rather than replaces, the existing [joint/risk comparison](related_work_constraints.md) and [deferral comparison](deferral_and_coverage_prior_work.md). Search results on structured QA routing and unrelated meanings of "trajectory" were not promoted to motion-prediction evidence.

## What These Results Require

**HCP.** Definition 1 requires permutation symmetry both across groups and within each group. Theorem 1 concerns marginal prediction-set coverage for an observation in a new group. Its assumptions do not follow from storing scene IDs. In equation (6), the test-group mass is `1/(K+1)` at infinity, where K is the number of calibration groups. Our direct calculation from this formula is below; it is not a new theorem. [Original text](https://arxiv.org/html/2306.06342v4).

**GHCP.** The reviewed construction uses labeled initial observations from the target group, alongside reference groups. A1 requires exchangeable group laws; A2 requires group-size ignorability; A3 requires conditional within-group IID observations and cross-group independence. It accommodates a different observation scheme, not arbitrary serial dependence. Applying it here would require an explicitly approved target-domain adaptation protocol and proof that its assumptions fit the sampling design. Counting past coordinates as if they were completed, independently labeled forecast examples would not suffice. [Original text, sections 1.1-2.2](https://arxiv.org/html/2608.15500v1).

**CAFHT.** Section 2.1 permits temporal dependence within each trajectory but assumes exchangeability across complete trajectories. Theorem 1 guarantees simultaneous marginal path coverage for fixed parameters. Section 3.4 separates parameter tuning from calibration or adjusts for selection. The online setup and multistep extension must be matched to the information available at each forecast origin; later observations cannot be supplied to a frozen-origin comparator. [Original text](https://arxiv.org/html/2402.09623v2).

**Adaptive motion-planning CP.** Corollary 3 bounds average one-step coverage over time. Theorem 7 connects this to average safety for a recursively feasible controller using the specified safety constraint. This is neither a per-scene ADE improvement theorem nor an assurance for our offline mixed forecasts. We have no corresponding closed-loop controller or verified physical geometry. [Original PDF](https://proceedings.mlr.press/v211/dixit23a/dixit23a.pdf).

## Analytical Checks

These are constructed examples, not estimates from SDD, UCY or DUT. They change no experiment configuration.

### Stationarity Does Not Establish Exchangeability

Consider a stationary, unit-variance Gaussian AR(1) sequence with correlation rho=0.8. Three consecutive observations have covariance

```text
          X1    X2    X3
X1       1.00  0.80  0.64
X2       0.80  1.00  0.80
X3       0.64  0.80  1.00
```

Swapping X2 and X3 changes Cov(X1,X2) from 0.80 to 0.64. The joint distribution is therefore not invariant to that permutation. Even stationarity plus a common recording identifier does not establish the required symmetry. This counterexample does not prove that every observed trajectory sample violates every conformal assumption; it disproves the shortcut "grouped time series implies exchangeability." Randomly shuffling storage order does not justify the law of a future, chronologically selected observation.

### More Rows Do Not Remove HCP's Group Mass

For finite calibration scores, HCP's finite mass is K/(K+1). The threshold is infinite when `1-alpha > K/(K+1)`, irrespective of observations per group. At hypothetical K=6, that mass is 6/7=0.857143: both 90% and 95% thresholds are infinite. Finite thresholds first become possible at K=9 and K=19 respectively. Equality can select the largest finite score. [Equation (6)](https://arxiv.org/html/2306.06342v4).

These are properties of this construction, not universal lower bounds. The project's six historical groups are not six approved calibration groups; actual formal support remains unresolved. An empirical interval from repeated scene resampling cannot manufacture new independent scenes.

### Coverage Does Not Order Forecast Accuracy

In a constructed one-dimensional prediction problem, let the true endpoint and baseline both be 0, while the candidate endpoint is 10. A candidate-centered interval [-1,21] covers the truth with probability one in this degenerate example, yet the candidate's absolute error exceeds the baseline's by 10. Thus coverage alone does not imply baseline-relative improvement. No real outcome is inferred from this example.

For our own notation, keep the following quantities separate:

| Quantity | Definition or interpretation | What it cannot establish alone |
| --- | --- | --- |
| Signed excess loss | `E[L(selected,Y) - L(baseline,Y)]` | No damage to any individual or subgroup |
| Positive-part harm | `E[max(L(selected,Y)-L(baseline,Y),0)]` | A relative easy-case percentage without its denominator |
| Clipped harm | Positive harm divided by a declared scale and capped at 1 | Control of unbounded raw error tails |
| Easy degradation | Ratio of aggregate selected and baseline errors on the prespecified easy slice | Per-person or physical safety |
| Intervention rate | Fraction of eligible agent queries switched | Equal realized risk or useful prediction gains |
| Prediction-set coverage | Probability that a target belongs to a returned set | Lower point-prediction error than another model |

## The Falsifiable Contribution

The working hypothesis is **not** that adding JEPA, a graph, an optimizer or a conformal label is novel. It is that baseline-relative supervision and joint intervention can improve the composition of fixed forecasts at a comparable intervention rate. The stronger statistical-risk claim is a separate, currently unsupported hypothesis.

| Question | Required controlled comparison | Result that would refute or narrow the claim | Current evidence |
| --- | --- | --- | --- |
| Do separate benefit/harm targets help? | Same OOF queries, forecasts, feature support and fitting budget: cost-sensitive deferral vs ridge vs neural cost heads | No reproducible advantage over the simpler head | Synthetic connection only; real comparison not_run |
| Does joint choice help beyond switching less? | Independent vs joint with the same actual count per query, retained unmatched-query ledger, nonzero-intervention support | Benefit disappears at matched counts or harms accuracy | Exact-count implementation checked; real comparison not_run |
| Does interaction information matter? | Same head and count control, interaction features/costs enabled vs disabled; report forecasting error and proximity separately | Only proxy collisions decrease while forecasting worsens | Formal ablation not_run |
| Does calibration add a supported guarantee? | Frozen family, independent units, prespecified bounded risk and confidence level, untouched confirmation | Unsupported sampling assumption, vacuous bound or zero useful intervention | Synthetic interface only; real calibration not_run |
| Does the effect generalize? | Frozen policies, matched public forecasters, at least three seeds and paired physical-scene uncertainty | Gain limited to an exposed recording or one seed | No clean confirmatory result |

Changing a head, training target or threshold after inspecting confirmation would make that data development material. Three seeds measure optimization variability, not new scene support. A gain under a shared budget cap is insufficient to attribute improvement to coordination if actual intervention rates differ.

## Consequences for the Current Study

1. Keep the paper title neutral until independent results support improvement and risk control.
2. Keep the existing empirical matched-control path. Do not replace it with an unvalidated HCP/GHCP wrapper to bypass the shortage of independent scenes.
3. If a prediction-band comparator is added, declare its output, forecast-origin information and scoring separately. Do not rank its coverage against point ADE as if they were the same task.
4. Treat new-site zero-shot evaluation, delayed-feedback online adaptation and labeled target-site adaptation as different studies. No target labels are newly authorized here.
5. Resolve the already-pending task/data-role and empirical-versus-formal-risk decisions before clean real fitting. Do not silently reuse historical test exposure as confirmation.

The shortest remaining empirical test is the already implemented, same-forecast OOF/development comparison under an approved protocol, followed by independently frozen confirmation. No further generic architecture expansion is justified by this review. There is no new predictive result, calibrated safety claim or deployment. Stage5C and SMC remain disabled; the work is not submission-ready.
