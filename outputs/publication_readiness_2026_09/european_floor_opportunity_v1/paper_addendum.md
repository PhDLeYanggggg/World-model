# Development Evidence: Reference-Action Consistency

## Motivation

A protected neural forecasting policy can be inferior to a strong protected
motion baseline even if its selected neural forecasts contain useful incremental
information. One source of this discrepancy is the action taken when neural
intervention is rejected. Evaluating against a strong baseline while reverting
to a weaker one confounds candidate quality, intervention quality and fallback
quality. We examine these components without changing the learned forecasts or
the saved intervention decisions.

## Diagnostic

For each row, let c, d and n denote the ADE of constant velocity, a frozen
protected-damping policy and the neural forecast, respectively. Let s be the
frozen causal intervention decision. We compare s*n+(1-s)*c with s*n+(1-s)*d.
The second expression is an offline rebasing diagnostic, not a risk-calibrated
deployment policy. We separately report realizable candidate opportunities
max(d-n,0), selected harm, missed benefit, and the signed cost of the default
action. Future errors are used only for retrospective accounting and oracle
upper bounds, never for intervention inputs.

## Findings and Limitations

Across both cross-moment variants, three seeds, three source-excluded folds and
two event targets, the 36 original all-ADE comparisons remain negative. Rebasing
the default action produces positive all-ADE gains of 0.1270--2.0713% over equally
protected damping, with positive conditional locality-bootstrap intervals in
all 36 views. Hard-subset points improve in all 36 views, with 34 positive intervals.
The positive pattern persists in complete-label and partial-label slices.

This does not establish safe neural superiority. Zero-reference cases are still
harmed in 24 repeated views, representing only four partially labeled rows from
one locality. Risk heads were not recalibrated relative to the new floor.
All twelve localities are already-opened development data; folds exclude eight
localities from the complete producer chain, but repeated seeds and intervals
are dependent and unadjusted. These are image-pixel detector-track forecasts,
not metric physical-world or independent-confirmation evidence.

The result motivates reference-consistent gain/harm learning with source-cross-
fitted floor targets and independent scene-level calibration. It does not yet
establish methodological novelty, a deployed neural world model, or a sufficient
main claim for submission. Stage5C and SMC remain disabled.
