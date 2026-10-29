# Why This Is Accounting, Not Another Hurdle Fit

## Verified Prior Records

- [Hurdle](../european_hurdle_risk_v1/failure_analysis.md): better occurrence
  scores did not establish safe intervention; zero-reference support was sparse.
- [Matched coverage](../european_hurdle_coverage_v1/support_v2/conclusions.md):
  changing the number of interventions explained important differences;
  hurdle ordering was not uniformly better at fixed counts.
- [Ranked hurdle](../european_ranked_hurdle_v1/failure_analysis.md): training a
  pairwise auxiliary did not reliably improve excluded-locality ordering;
  realized-label ratios and conditional moment ratios differ.
- [Task gradients](../european_task_gradients_v1/failure_analysis.md): broad
  gradient conflict was not a sufficient explanation; component effects varied.
- [Isolated harm](../european_isolated_harm_fp64_v1/failure_analysis.md): exact
  protection of benefit/reference outputs did not repair selected risk.

These studies use different cohorts/protocols. Their metrics must not be pooled
as one independent experiment. The present diagnosis reuses only the complete
isolated-harm fit/held grid and its frozen predictions for quantitative work.

## Relevant Original Literature

Hebert-Johnson et al., ICML2018, define calibration over a specified class of
computationally identifiable subpopulations rather than relying only on broad
averages. [Original paper and proceedings](https://proceedings.mlr.press/v80/hebert-johnson18a.html).

Jung et al., COLT2021, extend this framework to higher moments and describe
group-averaged uncertainty statements for sufficiently supported subgroups.
[Original proceedings](https://proceedings.mlr.press/v134/jung21a.html).

Relevance here is conceptual: a fixed-cohort average need not resolve the
population selected by the policy. We have NOT implemented either algorithm,
proved their assumptions, or obtained a multicalibration guarantee. Our causal
radius bins are descriptive diagnostics, not a new method. Bounded conditional
harm, repeated recordings, missing outcomes and policy selection require their
own analysis before any guarantee can be claimed.
