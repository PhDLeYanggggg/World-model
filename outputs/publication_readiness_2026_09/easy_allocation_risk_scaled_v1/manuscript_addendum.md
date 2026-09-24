# Manuscript Addendum: Scope of Query-Level Risk Allocation

## Method Control

With fixed forecasts, define predicted easy-weighted positive harm q_i, predicted
easy-weighted CV error r_i, and predicted net benefit g_i. Causal support and the
easy multiplier rho=0.02 are fixed. Compare pointwise protection with two
query-level binary allocations:

- Selected denominator: maximize sum x_i g_i subject to sum x_i(q_i-rho r_i)<=0.
- Population denominator: maximize sum x_i g_i subject to sum x_i q_i<=rho sum r_i.

The population denominator contains every forecastable target at that query,
including those kept at CV, but excludes non-target context and other times.
It is a distinct accounting relaxation. It must not be described as the same
selected-risk guarantee or attributed to joint-agent learning.

At the population allocation's cardinality, unary geometry removes only x_i*x_j
terms from the existing pairwise proximity objective; joint geometry retains
them. Both retain the same support, risk budget, forecasts and exact count.
These are conventional constrained optimization controls, not a novelty claim.

## Development Finding

Across four development-exposed SDD sites and three seeds, selected-set pooling
adds 0.003562 percentage points for Transformer over its failed pointwise gate.
Using the population denominator instead reaches 1.2802% equal-site ADE gain over
CV, versus 2.4368% under the previous strict neural rule. The population-minus-
strict paired-site interval is [-2.0064, -0.5223] percentage points. The new rule
reduces worst site/seed positive-easy degradation from 1.0669% to 0.1120%, but is not
a utility improvement at the already-satisfied 2% ceiling. Two complete zero-CV
query/seed cases are harmed and remain separately disclosed.

Nonadditive geometry changes unary choices in only 20 of 62,796 Transformer
query/seed instances, adding 0.00004811 percentage points of ADE gain, with
CI95 [0, 0.00009622]. This does not establish a practically meaningful interaction
contribution. Fewer proxy conflicts are not physical collision validation.

## Numerical Integrity

A cost-unit repair was required for 127 of 188,388 query/action/seed instances.
Original invalid solver proposals were rejected. An equivalent positive scaling
of all cost units preserves objectives and feasible sets; original-unit risk and
primal/dual checks remain mandatory. All other choices are unchanged, and full
metric reductions are rerun. Both versions and failure evidence are retained.

## Limits on Claims

Native 8/12 annotation steps, annotation pixels, only four explored sites,
overlapping windows, offline annotation provenance and incomplete outcomes.
Three-seed error means and site-level bootstrap are descriptive development
uncertainty, not independent confirmation. This study provides no new predictor,
no independently calibrated policy, no deployment promotion and no foundation,
true 3D, metric, seconds-level or physical-safety evidence. It does not establish
the proposed joint-agent mechanism as a paper's main contribution. It is useful
as a falsifiable negative control and a diagnosis of conditional-risk allocation.
Stage5C and SMC remain disabled.
