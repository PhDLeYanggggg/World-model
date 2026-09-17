# Conditional ADE Prediction Under Frozen Tree Partitions

## Material Passport

Mode: authorized fit-only mechanism experiment, not an independent evaluation.
The full eight-observed/twelve-predicted-step task, parent normalized-ADE metric,
fit scenes, target labels, static proxy features and historical exposure remain
unchanged. The pending native-ADE/FDE primary decision is not bypassed.

## Falsifiable Hypothesis

The corrected regressors were fitted using squared-error tree partitions and
return conditional mean coordinates. Is part of their unnecessary movement
caused by taking a mean rather than minimizing conditional Euclidean distance?
Change only the point decision under the SAME learned leaf distributions.
Do not search a new threshold, train a new classifier or claim a new architecture.

For each frozen unbootstrapped tree, a query weights each original training row
in its reached leaf uniformly. Average over256 trees. Verify that these weights
times original training labels reproduce the saved conditional mean forecast.
Derive one deterministic predicted point per future step as the weighted
geometric median of those same training labels. A conditional zero atom can
yield an exact zero prediction without an arbitrary movement threshold.

Use modified Weiszfeld updates, max5000, relative-to-support-scale objective-gap
tolerance1e-8; expose iteration-limit cases and gap bounds. Numerical tolerance
is not a deployment threshold. A minimum-norm subgradient and maximum distance
to training support bound the objective gap over its convex hull. This is an
implementation check, not a generalization or safety certificate.

The geometric-median algorithm is established, not a proposed M3W novelty:
Vardi and Zhang, PNAS97(4),1423-1426(2000),
[DOI10.1073/pnas.97.4.1423](https://doi.org/10.1073/pnas.97.4.1423).
Bibliographic metadata/abstract were verified via
[PubMed](https://pubmed.ncbi.nlm.nih.gov/10677477/); publisher/PMC full-text
access was unavailable in this session. This implementation is locally derived
and checked on convex examples, not represented as author code.

## Fixed Comparison

All18 ExtraTrees settings: ETH/Hotel held physical fit-scene folds, seeds17/29/43,
pooled/static-scene/directional-neighbor features. These are internal fit folds,
not new independent test scenes. Report all settings, with no winner promotion.
Compare unchanged mean, median, and each with the already frozen0.9 start gate.
Report native/parent ADE and FDE, zero-floor absolute easy harm, direction error,
intervention rate, run-balanced error and conditional training risk. No p-values
or independent-scene CI from two sites or overlapping windows.

The training labels define the conditional distribution only on the opposite
physical fit scene. Query weights use causal features only. Held labels enter
scoring after all forecasts are formed, never weights or medians. No goal
construction, new body/video features, hindsight magnitude filter or target
exclusion. Label-side quantization diagnostics are not inputs.

If medians reduce drift but retain zero/negative cross-site trajectory gains,
conclude that this point-decision repair alone is insufficient. If gains appear,
keep the all-setting/easy evidence and require a new prospective full-sample
study and independent support before deployment. This limited experiment is not
fresh neural training, latent generation, Stage5C, SMC or metric/seconds evidence.
