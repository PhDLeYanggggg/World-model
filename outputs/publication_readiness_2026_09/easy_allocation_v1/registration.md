# Same-Query Easy-Risk Allocation Experiment

Registered before new allocation decisions or their outcome readout, 2026-09-24.
Development only. All four SDD physical sites are design-exposed. This experiment
follows a completed negative pointwise easy-moment result; it is not an untouched
test, a new forecasting fit or an independent safety certificate.

## Question and fixed assets

Does query-level allocation recover useful interventions that the pointwise
positive-easy-harm condition discards? Does a nonadditive pair objective improve
over its unary decomposition at the same count and predicted risk budget?

Reuse all three frozen predictors, original neural benefit/harm heads and the
36 easy-moment forests from easy_moment_v1. No refitting. Four excluded sites,
three seeds, 33 recordings, all 175,756 past-eligible target windows. The producer
chain, shared easy/hard cutoffs, input schema, raw labels and primary metric are
unchanged. Rows with missing future labels remain eligible for inference.

## Decision rules

For each recording/frame query, q_i estimates positive easy-weighted harm and
r_i estimates easy-weighted baseline ADE, from the frozen forest. Only existing
net-positive, nonstationary target candidates may be selected. Non-target visible
context may affect the pair proxy, but supplies neither a learned risk allowance
nor an intervention variable. No decision pools across times, recordings, sites
or future visibility. The existing tolerance rho=0.02 stays fixed.

Report all following controls; do not select a winner on the outer results:

1. CV floor, original net-positive rule and old strict harm rule.
2. Pointwise: q_i <= rho*r_i for each switched target.
3. Aggregate-selected: maximize predicted net benefit subject to
   sum_i x_i*(q_i-rho*r_i) <= 0. This isolates pointwise versus selected-set risk.
4. Aggregate-population: maximize predicted net benefit subject to
   sum_i x_i*q_i <= rho*sum_i r_i over ALL forecastable targets in that query.
   The population denominator is a deliberate extra relaxation and is not
   silently attributed to joint geometry. Nonselected targets keep CV predictions.
5. Whole-scene uniform: switch every forecastable target or none, under the same
   population budget and net-positive support. If any target is unsupported, the
   common-support constraint makes the all-switch proposal inadmissible.
6. Aggregate-unary and aggregate-joint: use the population rule's realized
   cardinality, the same original budget/support and the pre-existing proximity
   table. Unary removes only pair-product terms. Joint retains them. Geometry
   weight1, radius=median past target scale, threshold=0.1 radius, as in the earlier
   registered joint study. No fitted pair weight or geometric threshold sweep.

The primary mechanism contrast is aggregate-selected minus pointwise, per
predictor. Secondary contrasts: aggregate-population minus aggregate-selected;
population minus old strict; joint minus unary; joint minus population at matched
counts. Query count matching is outcome-blind and does not imply matched realized
risk. Any numerical solver failure falls back to CV and remains in the report;
failure counts are not discarded. A failed matched solver invalidates its matched
contrast, not a reason to claim joint success. Use checked existing SciPy MILP
controls for population geometry; separately verify the signed selected budget.

## Evidence requirements

All decisions must be saved before reading the cached, already verified outcome
costs. The new allocation and reductions are fresh; reused predictor/cost/label
assets are cached-verified. No future array enters allocation. Retain all fixed
policies, site/seed results, available-point ADE, endpoint FDE, hard, positive-easy,
zero-CV, tails, unknown/partial label counts, full-grid bounds, intervention rates,
predicted budget usage and proximity changes. Report nonadditive opportunities
and decision changes, not just a solver objective. Predictions remain local
annotation pixels, obs8/pred12 native steps, not metric or seconds.

Use the existing equal-site relative ADE gain over CV as descriptive primary
metric, three seed means, and 3,000 paired physical-site bootstrap draws. Only
four explored sites support uncertainty; overlapping windows and seeds are not
new independent sites. Easy degradation must remain <=2% at every site/seed,
with zero-CV harms disclosed separately. Improving allocation gain while violating
easy protection does not count as a successful protected policy. No new deployment
follows even if an empirical contrast is favorable.

## Execution and boundaries

Native arm64 local CPU4/interop1/workers0, process lock, heartbeat, atomic query
chunks and resume. A fixed first256-query pilot estimates runtime; it is resumed
without changing the matrix. Exact decision/aggregate replay and separate
arithmetic/brute-force fixture verification are required. Slow computation is not
a reason to truncate the registered population. CREATE's last recorded access
failure does not establish queue status; this small local optimization needs no
new remote job. Runtime errors are recorded before any repair or resumption.

No threshold optimization, external prediction, independent calibration or
confirmation. DUT stays diagnostic; HT21 remains quarantined; DroneCrowd remains
closed. No physical safety, true3D, foundation or online annotation claim.
Stage5C and SMC remain off. Novelty and submission readiness are not established
by solving this conventional constrained allocation problem.
